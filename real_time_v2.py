#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실시간 키포인트 기반 수어 예측기(v2, 어텐션, 토르소 정규화 + 가변 길이)

- 웹캠 → MediaPipe Holistic → (토르소 정규화) 3-features(152) → GRU(v2)+Attention → 예측
- 길이 정책: FRAME_MIN(70) 이상이면 추론, 90 미만이면 현재 길이로, 90 이상이면 최근 90 프레임 사용
- CONF_THRESHOLD 이상일 때 npy 저장 (원본 구조 유지 아님; 수집용 경로에 카테고리/라벨/구간으로 분기)
"""

import os
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import pickle

from model_v2 import KeypointGRUModelV2

# =========================
# 0) 설정
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 폰트: NanumGothic 폴백 처리
def _load_font():
    candidates = [
        "NanumGothic.ttf",                       # 현재 폴더
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/Library/Fonts/AppleGothic.ttf",        # macOS
        "C:/Windows/Fonts/malgun.ttf",           # Windows
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, 24)
            except Exception:
                pass
    return ImageFont.load_default()

FONT = _load_font()

# 길이 정책
FRAME_MIN = 70
FRAME_TARGET = 90

# 경로/임계치
MODEL_DIR = "model"           # v2 학습 모델 폴더 (카테고리별 *_model.pth / *_label_map.pkl)
SAVE_DIR = "collected_data"   # 저장 루트
CONF_THRESHOLD = 90.0         # 90% 이상일 때만 저장
USE_CANON = True              # 토르소 정규화 on/off

# =========================
# 1) 카테고리 선택/모델 로드
# =========================
CATEGORY = input("테스트할 카테고리 이름을 입력하세요: ").strip()
label_map_path = os.path.join(MODEL_DIR, f"{CATEGORY}_label_map.pkl")
model_path = os.path.join(MODEL_DIR, f"{CATEGORY}_model.pth")

if not os.path.exists(label_map_path) or not os.path.exists(model_path):
    print(f"[ERR] 모델 또는 라벨맵 없음: {CATEGORY}")
    raise SystemExit(2)

with open(label_map_path, "rb") as f:
    label_map = pickle.load(f)
idx_to_label = {v: k for k, v in label_map.items()}

model = KeypointGRUModelV2(input_dim=152, attn_dim=146, num_classes=len(label_map)).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
print(f"[{CATEGORY}] v2 모델 로드 완료 — classes: {len(label_map)}  device: {DEVICE}")

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
# 3) 토르소 정규화 유틸
# =========================
POSE_L_SHOULDER, POSE_R_SHOULDER = 11, 12
POSE_L_HIP, POSE_R_HIP = 23, 24

def safe_unit(v, eps=1e-8):
    n = np.linalg.norm(v)
    return v / (n + eps)

def build_torso_frame(pose_33x4):
    """
    포즈에서 x=우향(오른어깨-왼어깨), y=상향(어깨중심-엉덩이중심), z=x×y
    원점=mid_hip, scale=어깨폭
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
    # 직교화
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
        Q = Q / scale
        return Q.astype(np.float32)

    pose_xyz = pose_33x4[:, :3]
    pose_vis = pose_33x4[:, 3:4]
    pose_can = np.concatenate([xf(pose_xyz), pose_vis], axis=1).astype(np.float32)
    return pose_can, xf(face_468x3), xf(lh_21x3), xf(rh_21x3)

# =========================
# 4) 3-features 추출
# =========================
def calculate_relative_hand_coords(hand_kpts):
    if hand_kpts.size == 0 or np.all(hand_kpts == 0):
        return hand_kpts
    wrist = hand_kpts[0]
    return (hand_kpts - wrist).astype(np.float32)

def calculate_finger_angles(hand_kpts):
    if hand_kpts.size == 0 or np.all(hand_kpts == 0):
        return np.zeros(10, dtype=np.float32)
    angles = []
    idxs = {
        'thumb':[1,2,3,4], 'index':[5,6,7,8], 'middle':[9,10,11,12],
        'ring':[13,14,15,16], 'pinky':[17,18,19,20]
    }
    for js in idxs.values():
        for i in range(len(js)-2):
            p1, p2, p3 = hand_kpts[js[i]], hand_kpts[js[i+1]], hand_kpts[js[i+2]]
            v1, v2 = p1 - p2, p3 - p2
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            c = float(np.dot(v1, v2) / denom)
            c = max(min(c, 1.0), -1.0)
            angles.append(float(np.arccos(c)))
    return np.array(angles, dtype=np.float32)

def calculate_hand_face_relation(lh_kpts, rh_kpts, face_kpts):
    nose = face_kpts[1] if face_kpts.size != 0 and np.any(face_kpts) else np.zeros(3, dtype=np.float32)
    lw = lh_kpts[0] if lh_kpts.size != 0 and np.any(lh_kpts) else np.zeros(3, dtype=np.float32)
    rw = rh_kpts[0] if rh_kpts.size != 0 and np.any(rh_kpts) else np.zeros(3, dtype=np.float32)
    return np.concatenate([lw - nose, rw - nose]).astype(np.float32)

def extract_feature(landmarks):
    """Holistic 결과 → (152,) 특징. USE_CANON일 때 토르소 정규화 후 계산."""
    pose = np.array([[l.x, l.y, l.z, l.visibility] for l in landmarks.pose_landmarks.landmark], dtype=np.float32) if landmarks.pose_landmarks else np.zeros((33, 4), dtype=np.float32)
    face = np.array([[l.x, l.y, l.z] for l in landmarks.face_landmarks.landmark], dtype=np.float32) if landmarks.face_landmarks else np.zeros((468, 3), dtype=np.float32)
    lh   = np.array([[l.x, l.y, l.z] for l in landmarks.left_hand_landmarks.landmark], dtype=np.float32) if landmarks.left_hand_landmarks else np.zeros((21, 3), dtype=np.float32)
    rh   = np.array([[l.x, l.y, l.z] for l in landmarks.right_hand_landmarks.landmark], dtype=np.float32) if landmarks.right_hand_landmarks else np.zeros((21, 3), dtype=np.float32)

    if USE_CANON:
        pose, face, lh, rh = canonicalize(pose, face, lh, rh)

    relative_lh = calculate_relative_hand_coords(lh).reshape(-1)  # 63
    relative_rh = calculate_relative_hand_coords(rh).reshape(-1)  # 63
    angles_lh = calculate_finger_angles(lh)                       # 10
    angles_rh = calculate_finger_angles(rh)                       # 10
    rel_feat = calculate_hand_face_relation(lh, rh, face)         # 6

    feat152 = np.concatenate([relative_lh, relative_rh, angles_lh, angles_rh, rel_feat]).astype(np.float32)
    return feat152, lh, rh

# =========================
# 5) 렌더링/저장 유틸
# =========================
def draw_text(img, text, position, color=(255, 0, 255)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=FONT, fill=color)
    return np.array(img_pil)

def save_sequence(sequence, label, confidence):
    # 90_95 / 95_100 / 100
    if confidence < 95:
        range_folder = "90_95"
    elif confidence < 100:
        range_folder = "95_100"
    else:
        range_folder = "100"

    save_path = os.path.join(SAVE_DIR, CATEGORY, label, range_folder)
    os.makedirs(save_path, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{int(confidence)}_{timestamp}.npy"
    np.save(os.path.join(save_path, filename), np.array(sequence, dtype=np.float32))
    print(f"[SAVE] {save_path}/{filename} (T={len(sequence)})")

# =========================
# 6) 실시간 루프
# =========================
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=FRAME_TARGET)  # 최근 90으로 제한
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

        if collecting:
            feat, lh, rh = extract_feature(results)
            if np.any(lh) or np.any(rh):
                hand_detected_any = True
            sequence.append(feat)

            T = len(sequence)
            info = f"{T}/{FRAME_TARGET} 수집 중..."
            frame = draw_text(frame, info, (30, 30))

            # 적응형 추론: 최소 70 도달 시 추론 수행
            if T >= FRAME_MIN:
                # 입력 준비: T' = T (<=90) 또는 최근 90
                if T > FRAME_TARGET:
                    xs = list(sequence)[-FRAME_TARGET:]
                    lengths = torch.tensor([FRAME_TARGET], dtype=torch.long, device=DEVICE)
                else:
                    xs = list(sequence)
                    lengths = torch.tensor([T], dtype=torch.long, device=DEVICE)

                xb = torch.tensor(np.array(xs, dtype=np.float32), dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1,T',152]

                if not hand_detected_any:
                    result_text = "검출된 손 없음"
                else:
                    with torch.no_grad():
                        logits = model(xb, lengths)     # v2: (x, lengths)
                        prob = F.softmax(logits, dim=-1)  # [1,C]
                        pred = int(prob.argmax(dim=-1).item())
                        confidence = float(prob[0, pred].item() * 100.0)
                        label = idx_to_label[pred]
                        result_text = f"{label} ({confidence:.1f}%)"

                        if confidence >= CONF_THRESHOLD:
                            save_sequence(xs, label, confidence)

                # 한 번 예측 후 상태 초기화(원하면 지속 예측 모드로 변경 가능)
                collecting = False
                sequence.clear()
                hand_detected_any = False

        # 상태 텍스트
        frame = draw_text(frame, result_text, (30, 30))
        cv2.imshow("Real-time Sign Recognition (v2)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            collecting = True
            sequence.clear()
            hand_detected_any = False
            result_text = ""
        elif key == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
