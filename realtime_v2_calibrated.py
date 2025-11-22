#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실시간 키포인트 기반 수어 예측기(v2, 어텐션, 토르소 정규화 + 가변 길이)
+ Calibration(T, β) + 5단계 임계(클래스별) + 마진 규칙

- 웹캠 → MediaPipe Holistic → 3-features(152, 토르소 정규화) → GRU(v2)+Attention
- 길이 정책: FRAME_MIN(70) 이상이면 추론, 90 미만이면 현재 길이로, 90 이상이면 최근 90 프레임 사용
- 보정: softmax(z/T) 후 top-1 p에 지수 보정 f(p;β) 적용 (단조이므로 순위 불변)
- 의사결정: 클래스별 τ₁..τ₄ + 마진 m₁..m₅으로 L1~L5 결정
- 저장: L4 → 95_100, L5 → 100 (L3 이하는 저장 안 함)

- 추가(모델 관점 피드백 + 숫자키 선택 + 등급):
  - 실행 시 CATEGORY(예: Day1)만 입력
  - 화면 상단에 현재 "선택 단어"가 표시됨
  - 1: 이전 단어, 2: 다음 단어, 's': 그 단어로 시작
  - 타깃 단어에 대한 보정 확률 p_target' 기준:
      · p ≥ 0.95  → Perfect (피드백 X)
      · 0.70–0.95 → OK (피드백 O)
      · 0.55–0.70 → Not Bad (피드백 O)
      · p < 0.55  → Bad + 틀렸다고 안내 (피드백 X)
"""

import os, time, json
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import pickle

from model_v2 import KeypointGRUModelV2
from model_feedback_groups import compute_group_importance

# =========================
# 0) 설정
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_font():
    candidates = [
        "NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/Library/Fonts/AppleGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                # 글자 크기 18
                return ImageFont.truetype(p, 12)
            except Exception:
                pass
    return ImageFont.load_default()

FONT = _load_font()

# 길이 정책
FRAME_MIN = 90
FRAME_TARGET = 90

# 경로/설정
MODEL_DIR = "model"
SAVE_DIR  = "collected_data"
USE_CANON = True  # 토르소 정규화

# 등급 기준 (보정 확률 p_target')
THR_BAD      = 0.55   # 미만이면 Bad (틀림)
THR_NOTBAD   = 0.55   # 이상
THR_OK       = 0.70   # 이상
THR_PERFECT  = 0.95   # 이상

# =========================
# 1) 카테고리 선택/모델 & 보정 로드
# =========================
CATEGORY = input("테스트할 카테고리 이름을 입력하세요: ").strip()
label_map_path = os.path.join(MODEL_DIR, f"{CATEGORY}_label_map.pkl")
model_path     = os.path.join(MODEL_DIR, f"{CATEGORY}_model.pth")
calib_path     = os.path.join(MODEL_DIR, f"{CATEGORY}_calib.json")

if not os.path.exists(label_map_path) or not os.path.exists(model_path):
    print(f"[ERR] 모델 또는 라벨맵 없음: {CATEGORY}")
    raise SystemExit(2)
if not os.path.exists(calib_path):
    print(f"[ERR] 보정 파일 없음: {calib_path}  (먼저 calibrate_and_export.py 실행)")
    raise SystemExit(2)

with open(label_map_path, "rb") as f:
    label_map = pickle.load(f)
idx_to_label = {v: k for k, v in label_map.items()}

# 라벨을 이름 순으로 정렬해서 고정 순서로 사용
labels_list = sorted(label_map.keys())
if not labels_list:
    print("[ERR] 라벨이 비어 있습니다.")
    raise SystemExit(2)

# 현재 선택된 타깃 단어 (숫자키로 움직임)
selected_idx = 0
current_target_label = labels_list[selected_idx]
current_target_idx   = label_map[current_target_label]

with open(calib_path, "r", encoding="utf-8") as f:
    calib = json.load(f)
T     = float(calib["temperature"])
BETA  = float(calib["beta"])
TAU   = {lab: v["tau"] for lab, v in calib["thresholds"].items()}  # {'label':[t1..t4]}
MARG  = calib["margins"]  # {'L1':..., 'L2':...}
M1 = float(MARG["L1"]); M2 = float(MARG["L2"]); M3 = float(MARG["L3"]); M4 = float(MARG["L4"]); M5 = float(MARG["L5"])

model = KeypointGRUModelV2(input_dim=152, attn_dim=146, num_classes=len(label_map)).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
print(f"[{CATEGORY}] v2 모델 + Calibration 로드 완료 — classes: {len(label_map)}  device: {DEVICE}")
print(f"  Temperature T={T:.3f}, Beta={BETA:.3f}")
print(f"  등급 기준: Bad<{THR_BAD*100:.0f}% < NotBad/OK < {THR_PERFECT*100:.0f}%≤Perfect")
print(f"  초기 선택 단어: {current_target_label}")
print("  키 사용법: 1=이전 단어, 2=다음 단어, s=시작, q=종료")

# =========================
# 2) MediaPipe 초기화
# =========================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# =========================
# 3) 토르소 정규화 / 3-features 추출 유틸
# =========================

POSE_L_SHOULDER = 11
POSE_R_SHOULDER = 12
POSE_L_HIP      = 23
POSE_R_HIP      = 24

def safe_unit(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n

def build_torso_frame(pose_33x4):
    """
    Mediapipe Pose 33*4 (x,y,z,visibility)에서
    어깨+엉덩이를 이용해 몸통(토르소) 기준 좌표계 구성
    """
    try:
        l_sh = pose_33x4[POSE_L_SHOULDER, :3]
        r_sh = pose_33x4[POSE_R_SHOULDER, :3]
        l_hp = pose_33x4[POSE_L_HIP, :3]
        r_hp = pose_33x4[POSE_R_HIP, :3]
    except Exception:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0, False

    if np.all(l_sh == 0) or np.all(r_sh == 0) or np.all(l_hp == 0) or np.all(r_hp == 0):
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0, False

    mid_hip = 0.5 * (l_hp + r_hp)
    mid_sh  = 0.5 * (l_sh + r_sh)

    x_axis = safe_unit(r_sh - l_sh)
    y_axis = safe_unit(mid_sh - mid_hip)
    y_axis = safe_unit(y_axis - np.dot(y_axis, x_axis) * x_axis)
    z_axis = safe_unit(np.cross(x_axis, y_axis))
    x_axis = safe_unit(np.cross(y_axis, z_axis))
    y_axis = safe_unit(np.cross(z_axis, x_axis))
    R = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
    scale = max(np.linalg.norm(r_sh - l_sh), 1e-3)
    return R, mid_hip.astype(np.float32), float(scale), True

def canonicalize(pose_33x4, face_468x3, lh_21x3, rh_21x3):
    R, origin, scale, ok = build_torso_frame(pose_33x4)
    if not ok:
        return pose_33x4, face_468x3, lh_21x3, rh_21x3

    def xf(P):
        if P.size == 0:
            return P
        Q = (P - origin[None, :]) @ R
        return Q / scale

    pose_c = xf(pose_33x4[:, :3])
    face_c = xf(face_468x3)
    lh_c   = xf(lh_21x3)
    rh_c   = xf(rh_21x3)
    return pose_c, face_c, lh_c, rh_c

def landmarks_to_np(results):
    """
    MediaPipe Holistic 결과에서 pose, face, left hand, right hand를 numpy로 변환
    """
    pose_33x4  = np.zeros((33, 4), dtype=np.float32)
    face_468x3 = np.zeros((468, 3), dtype=np.float32)
    lh_21x3    = np.zeros((21, 3), dtype=np.float32)
    rh_21x3    = np.zeros((21, 3), dtype=np.float32)

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            if i < 33:
                pose_33x4[i, 0] = lm.x
                pose_33x4[i, 1] = lm.y
                pose_33x4[i, 2] = lm.z
                pose_33x4[i, 3] = lm.visibility

    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark):
            if i < 468:
                face_468x3[i, 0] = lm.x
                face_468x3[i, 1] = lm.y
                face_468x3[i, 2] = lm.z

    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            if i < 21:
                lh_21x3[i, 0] = lm.x
                lh_21x3[i, 1] = lm.y
                lh_21x3[i, 2] = lm.z

    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            if i < 21:
                rh_21x3[i, 0] = lm.x
                rh_21x3[i, 1] = lm.y
                rh_21x3[i, 2] = lm.z

    return pose_33x4, face_468x3, lh_21x3, rh_21x3

# ====== 3-features(152D) 계산 관련 함수들 ======

def calculate_relative_hand_coords(lh_21x3, rh_21x3):
    # 21 * 3 = 63, 왼손/오른손 각각
    # 손목(0번)을 기준으로 상대 좌표
    def rel_hand(hand):
        if np.all(hand == 0):
            return np.zeros_like(hand)
        wrist = hand[0:1, :]
        return hand - wrist

    lh_rel = rel_hand(lh_21x3)
    rh_rel = rel_hand(rh_21x3)

    # (63,) + (63,) = (126,)
    return lh_rel.reshape(-1), rh_rel.reshape(-1)

def calculate_finger_angles(hand_21x3):
    # TODO: 실제 각도 계산은 기존 전처리 코드와 맞게 구현되어 있어야 함
    if np.all(hand_21x3 == 0):
        return np.zeros(10, dtype=np.float32)
    return np.zeros(10, dtype=np.float32)

def calculate_hand_face_relation(lh_21x3, rh_21x3, face_468x3):
    # 얼굴 기준 (예: 코 좌표) 대비 양 손목 거리 등 6차원 피쳐
    if np.all(face_468x3 == 0):
        return np.zeros(6, dtype=np.float32)

    nose = face_468x3[1, :]  # face mesh 코 tip index (기존 코드와 동일해야 함)
    lh_wrist = lh_21x3[0, :] if not np.all(lh_21x3 == 0) else nose
    rh_wrist = rh_21x3[0, :] if not np.all(rh_21x3 == 0) else nose

    vec_l = lh_wrist - nose
    vec_r = rh_wrist - nose

    dist_l = np.linalg.norm(vec_l)
    dist_r = np.linalg.norm(vec_r)

    feat = np.concatenate([vec_l, vec_r, [dist_l, dist_r]], axis=0)
    if feat.shape[0] > 6:
        feat = feat[:6]
    elif feat.shape[0] < 6:
        feat = np.pad(feat, (0, 6 - feat.shape[0]))
    return feat.astype(np.float32)

def extract_feature(results):
    pose_33x4, face_468x3, lh_21x3, rh_21x3 = landmarks_to_np(results)

    if USE_CANON:
        pose_c, face_c, lh_c, rh_c = canonicalize(pose_33x4, face_468x3, lh_21x3, rh_21x3)
    else:
        pose_c, face_c, lh_c, rh_c = pose_33x4[:, :3], face_468x3, lh_21x3, rh_21x3

    relative_lh, relative_rh = calculate_relative_hand_coords(lh_c, rh_c)  # 63 + 63
    angles_lh = calculate_finger_angles(lh_c)                               # 10
    angles_rh = calculate_finger_angles(rh_c)                               # 10
    rel_feat  = calculate_hand_face_relation(lh_c, rh_c, face_c)           # 6

    feat152 = np.concatenate([relative_lh, relative_rh, angles_lh, angles_rh, rel_feat]).astype(np.float32)
    return feat152, lh_c, rh_c

# =========================
# 5) 보정/단계결정/렌더/저장
# =========================
def softmax_T(logits, T):
    return F.softmax(logits / T, dim=-1)

def exp_squash(p, beta):
    # p in [0,1]
    eb = np.exp(beta)
    num = np.exp(beta * p) - 1.0
    den = eb - 1.0 + 1e-12
    return float(np.clip(num / den, 0.0, 1.0))

def group_name_ko(name: str) -> str:
    """그룹 이름을 한국어 설명으로 변환"""
    if name == "finger_coord":
        return "손 위치와 전체 자세"
    if name == "finger_angle":
        return "손가락 모양"
    if name == "nose_hand":
        return "얼굴 대비 손 위치"
    return name

def decide_level(label, p1p, p2p):
    """label(str), p1p=p'(top1), p2p=p'(top2) -> (L, save_bucket)"""
    tau1, tau2, tau3, tau4 = TAU.get(label, [1.01,1.01,1.01,1.01])
    gap = p1p - p2p
    # L5
    if (p1p >= tau4) and (gap >= M5): return "L5", "100"
    # L4
    if (p1p >= tau3) and (gap >= M4): return "L4", "95_100"
    # L3
    if (p1p >= tau2) and (gap >= M3): return "L3", None
    # L2
    if (p1p >= tau1) and (gap >= M2): return "L2", None
    # L1
    if gap >= M1: return "L1", None
    return "L1", None

def draw_text(img, text, position, color=(255, 0, 255)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=FONT, fill=color)
    return np.array(img_pil)

def save_sequence(sequence, label, level, bucket):
    """
    bucket:
      - "100" → 100점 버킷
      - "95_100" → 95~100 버킷
      - None → 저장 안함
    """
    if bucket is None:
        return
    save_path = os.path.join(SAVE_DIR, CATEGORY, bucket, label)
    os.makedirs(save_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}_L{level}.npy"
    np.save(os.path.join(save_path, filename), np.array(sequence, dtype=np.float32))
    print(f"[SAVE] {save_path}/{filename} (T={len(sequence)})")

# =========================
# 6) 실시간 루프
# =========================
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=FRAME_TARGET)
collecting = False
hand_detected_any = False
result_text = "Press 's' to start"

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)

        # ---- 선택 단어 안내 (항상 상단에 표시) ----
        select_msg = f"선택 단어: '{current_target_label}'  (1: 이전, 2: 다음, s: 시작, q: 종료)"
        frame = draw_text(frame, select_msg, (30, 20))

        if collecting:
            feat, lh, rh = extract_feature(results)
            if np.any(lh) or np.any(rh):
                hand_detected_any = True
            sequence.append(feat)

            Tcur = len(sequence)
            info = f"{Tcur}/{FRAME_TARGET} 수집 중..."
            frame = draw_text(frame, info, (30, 50))

            if Tcur >= FRAME_MIN:
                # 입력 길이 구성
                if Tcur > FRAME_TARGET:
                    xs = list(sequence)[-FRAME_TARGET:]
                    length = FRAME_TARGET
                else:
                    xs = list(sequence)
                    length = Tcur

                xs_array = np.array(xs, dtype=np.float32)
                xb = torch.tensor(xs_array, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                L  = torch.tensor([length], dtype=torch.long, device=DEVICE)

                if not hand_detected_any:
                    result_text = "검출된 손 없음"
                else:
                    with torch.no_grad():
                        logits = model(xb, L)               # [1,C]
                        qT = softmax_T(logits, T=T)         # [1,C]
                        qT_np = qT.squeeze(0).cpu().numpy()
                        top1 = int(qT_np.argmax())
                        p1 = float(qT_np[top1])
                        p2 = float(np.partition(qT_np, -2)[-2])

                        # 지수 보정 (top1 기준, 저장용)
                        p1p = exp_squash(p1, BETA)
                        p2p = exp_squash(p2, BETA)

                        label = idx_to_label[top1]
                        level, bucket = decide_level(label, p1p, p2p)

                        # 저장 정책 (기존 기준 유지)
                        save_sequence(xs, label, level, bucket)

                        # 게임용: 현재 선택된 타깃 단어 기준 유사도 판단
                        if current_target_label is not None and current_target_idx is not None:
                            p_target_raw = float(qT_np[current_target_idx])
                            p_target_p   = exp_squash(p_target_raw, BETA)  # 0~1
                            score_str = f"{p_target_p*100:.1f}%"

                            if p_target_p >= THR_PERFECT:
                                # PERFECT: 피드백 X
                                result_text = (
                                    f"Perfect! [{current_target_label}] {score_str} "
                                    "거의 완벽하게 표현했어요!"
                                )

                            elif p_target_p >= THR_OK:
                                # OK: 피드백 O
                                group_imp = compute_group_importance(
                                    model=model,
                                    xs_array=xs_array,
                                    length=length,
                                    class_idx=current_target_idx,
                                    device=DEVICE,
                                )
                                weak_sorted = sorted(group_imp.items(), key=lambda x: x[1])
                                if len(weak_sorted) >= 2:
                                    g1, g2 = weak_sorted[0][0], weak_sorted[1][0]
                                    desc = f"{group_name_ko(g1)}와(과) {group_name_ko(g2)}"
                                else:
                                    g1 = weak_sorted[0][0]
                                    desc = group_name_ko(g1)

                                result_text = (
                                    f"OK [{current_target_label}] {score_str} "
                                    f"전반적으로 잘 했어요. "
                                    f"다음엔 {desc} 쪽을 조금 더 의식해 보면 좋아요."
                                )

                            elif p_target_p >= THR_NOTBAD:
                                # Not Bad: 피드백 O
                                group_imp = compute_group_importance(
                                    model=model,
                                    xs_array=xs_array,
                                    length=length,
                                    class_idx=current_target_idx,
                                    device=DEVICE,
                                )
                                weak_sorted = sorted(group_imp.items(), key=lambda x: x[1])
                                if len(weak_sorted) >= 2:
                                    g1, g2 = weak_sorted[0][0], weak_sorted[1][0]
                                    desc = f"{group_name_ko(g1)}와(과) {group_name_ko(g2)}"
                                else:
                                    g1 = weak_sorted[0][0]
                                    desc = group_name_ko(g1)

                                result_text = (
                                    f"Not Bad [{current_target_label}] {score_str} "
                                    f"기본적인 형태는 비슷해요. "
                                    f"특히 {desc} 부분을 더 또렷하게 해보면 더 좋아질 것 같아요."
                                )

                            else:
                                # Bad: 틀렸다고 안내, 피드백 X
                                result_text = (
                                    f"Bad... [{current_target_label}] {score_str} "
                                    "아직 목표 수어와 차이가 커요. "
                                    "한 번 더 천천히 따라 해볼까요?"
                                )
                        else:
                            # 타깃이 없으면 기존 top-1 표시
                            result_text = f"{label} {level} ({p1p*100:.1f}%)"

                # 한 번 예측 후 초기화
                collecting = False
                sequence.clear()
                hand_detected_any = False

        # 피드백 텍스트 (두 번째 줄)
        frame = draw_text(frame, result_text, (30, 80))
        cv2.imshow("Real-time Sign Recognition (v2, Calibrated)", frame)

        # 키 입력 처리 (숫자 + 문자)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            collecting = True
            sequence.clear()
            hand_detected_any = False
            result_text = ""  # 새 라운드 시작
        elif key == ord('1'):
            # 이전 단어
            selected_idx = (selected_idx - 1) % len(labels_list)
            current_target_label = labels_list[selected_idx]
            current_target_idx   = label_map[current_target_label]
            result_text = f"단어 변경: '{current_target_label}'"
        elif key == ord('2'):
            # 다음 단어
            selected_idx = (selected_idx + 1) % len(labels_list)
            current_target_label = labels_list[selected_idx]
            current_target_idx   = label_map[current_target_label]
            result_text = f"단어 변경: '{current_target_label}'"

finally:
    cap.release()
    cv2.destroyAllWindows()
