#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
既存 holistic(.npy) → (Torso-Canonical) 3-features(152D) 변환기

입력 형식: 각 프레임이 1D 벡터이며, 다음 순서로 이어진 형태라고 가정
  pose: (33,4)  -> x,y,z,visibility  => 132
  face: (468,3) -> x,y,z             => 1404
  left hand: (21,3)                  => 63
  right hand: (21,3)                 => 63
총 길이 = 1662

출력: (T,152)
  - 상대 손 좌표(왼/오른손, 손목 기준) 63×2
  - 손가락 관절각 10×2
  - 얼굴(코=face index 1)→양 손목 벡터 6

옵션:
  --no-canon    토르소(어깨/엉덩이 기반) 정규화 비활성화
  --in/--out    단일 파일 변환(시험용)
"""

import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

# ------- 상수 -------
POSE_L_SHOULDER = 11
POSE_R_SHOULDER = 12
POSE_L_HIP = 23
POSE_R_HIP = 24

HOLISTIC_DIM = 33*4 + 468*3 + 21*3 + 21*3  # 1662

# ------- 유틸 -------
def list_npy_recursively(root: str) -> List[str]:
    out = []
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".npy"):
                out.append(os.path.join(d, f))
    out.sort()
    return out

def ensure_parent(p: str) -> None:
    pd = os.path.dirname(p)
    if pd and not os.path.exists(pd):
        os.makedirs(pd, exist_ok=True)

def safe_unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

def build_torso_frame(pose_33x4: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """토르소 좌표계(R, origin, scale, ok)"""
    try:
        l_sh = pose_33x4[POSE_L_SHOULDER, :3]
        r_sh = pose_33x4[POSE_R_SHOULDER, :3]
        l_hp = pose_33x4[POSE_L_HIP, :3]
        r_hp = pose_33x4[POSE_R_HIP, :3]
    except Exception:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0, False

    if (np.all(l_sh==0) or np.all(r_sh==0) or np.all(l_hp==0) or np.all(r_hp==0)):
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0, False

    mid_hip = 0.5*(l_hp + r_hp)
    mid_sh  = 0.5*(l_sh + r_sh)

    x_axis = safe_unit(r_sh - l_sh)                  # 오른쪽(+x)
    y_axis = safe_unit(mid_sh - mid_hip)             # 위쪽(+y)
    y_axis = safe_unit(y_axis - np.dot(y_axis, x_axis)*x_axis)   # x에 직교화
    z_axis = safe_unit(np.cross(x_axis, y_axis))     # 전방(+z)
    x_axis = safe_unit(np.cross(y_axis, z_axis))     # 재직교
    y_axis = safe_unit(np.cross(z_axis, x_axis))

    R = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)  # 3x3
    scale = float(max(np.linalg.norm(r_sh - l_sh), 1e-3))              # 어깨폭
    return R, mid_hip.astype(np.float32), scale, True

def canonicalize(pose_33x4: np.ndarray,
                 face_468x3: np.ndarray,
                 lh_21x3: np.ndarray,
                 rh_21x3: np.ndarray,
                 use_canon: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not use_canon:
        return pose_33x4, face_468x3, lh_21x3, rh_21x3

    R, origin, scale, ok = build_torso_frame(pose_33x4)
    if not ok:
        return pose_33x4, face_468x3, lh_21x3, rh_21x3

    def xf(P):
        if P.size == 0: return P
        Q = (P - origin[None, :]) @ R
        Q = Q / scale
        return Q.astype(np.float32)

    pose_xyz = pose_33x4[:, :3]
    pose_vis = pose_33x4[:, 3:4]
    pose_can = np.concatenate([xf(pose_xyz), pose_vis], axis=1).astype(np.float32)
    return pose_can, xf(face_468x3), xf(lh_21x3), xf(rh_21x3)

# ------- 3-features -------
def rel_hand(hand_21x3: np.ndarray) -> np.ndarray:
    if hand_21x3.size == 0 or np.all(hand_21x3==0):
        return hand_21x3
    wrist = hand_21x3[0]
    return (hand_21x3 - wrist).astype(np.float32)

def finger_angles(hand_21x3: np.ndarray) -> np.ndarray:
    if hand_21x3.size == 0 or np.all(hand_21x3==0):
        return np.zeros(10, dtype=np.float32)
    idxs = {
        "thumb":[1,2,3,4],
        "index":[5,6,7,8],
        "middle":[9,10,11,12],
        "ring":[13,14,15,16],
        "pinky":[17,18,19,20],
    }
    out = []
    for js in idxs.values():
        for i in range(len(js)-2):
            p1, p2, p3 = hand_21x3[js[i]], hand_21x3[js[i+1]], hand_21x3[js[i+2]]
            v1, v2 = p1-p2, p3-p2
            denom = (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
            c = float(np.dot(v1, v2)/denom)
            c = max(min(c, 1.0), -1.0)
            out.append(float(np.arccos(c)))
    return np.array(out, dtype=np.float32)

def face_to_wrists(lh_21x3: np.ndarray, rh_21x3: np.ndarray, face_468x3: np.ndarray) -> np.ndarray:
    nose = face_468x3[1] if face_468x3.size!=0 and np.any(face_468x3) else np.zeros(3, dtype=np.float32)
    lw = lh_21x3[0] if lh_21x3.size!=0 and np.any(lh_21x3) else np.zeros(3, dtype=np.float32)
    rw = rh_21x3[0] if rh_21x3.size!=0 and np.any(rh_21x3) else np.zeros(3, dtype=np.float32)
    return np.concatenate([lw - nose, rw - nose]).astype(np.float32)

def features_from_seq(holistic_seq: np.ndarray, use_canon: bool) -> np.ndarray:
    if holistic_seq.ndim != 2 or holistic_seq.shape[1] != HOLISTIC_DIM:
        raise ValueError(f"Expected shape (T,{HOLISTIC_DIM}), got {holistic_seq.shape}")
    feats = []
    for frame in holistic_seq:
        pose = frame[0:33*4].reshape(33,4)
        face = frame[33*4:33*4+468*3].reshape(468,3)
        lh   = frame[33*4+468*3:33*4+468*3+21*3].reshape(21,3)
        rh   = frame[33*4+468*3+21*3:].reshape(21,3)

        pose, face, lh, rh = canonicalize(pose, face, lh, rh, use_canon)

        rel_l = rel_hand(lh).reshape(-1)  # 63
        rel_r = rel_hand(rh).reshape(-1)  # 63
        ang_l = finger_angles(lh)         # 10
        ang_r = finger_angles(rh)         # 10
        f2w   = face_to_wrists(lh, rh, face)  # 6

        feat = np.concatenate([rel_l, rel_r, ang_l, ang_r, f2w]).astype(np.float32)  # 152
        feats.append(feat)
    return np.stack(feats).astype(np.float32)

# ------- 메인 로직 -------
def process_one(in_path: str, out_path: str, use_canon: bool, overwrite: bool) -> Tuple[bool, str]:
    if (not overwrite) and os.path.exists(out_path):
        return True, f"Skip (exists): {out_path}"
    try:
        seq = np.load(in_path)
        if seq.size == 0:
            ensure_parent(out_path)
            np.save(out_path, seq)
            return True, f"Empty seq saved: {out_path}"
        feats = features_from_seq(seq, use_canon)
        ensure_parent(out_path)
        np.save(out_path, feats)
        return True, f"OK: {out_path}  (T={feats.shape[0]}, D=152)"
    except Exception as e:
        return False, f"FAIL: {in_path} → {e}"

def parse_args(argv: list) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Existing holistic(.npy) → 3-features(152D)")
    p.add_argument("--keypoints-dir", default=None, help="holistic npy root (recursive). Use with --features-dir.")
    p.add_argument("--features-dir", default=None, help="output root for features (mirror structure).")
    p.add_argument("--in", dest="in_path", default=None, help="single input .npy path")
    p.add_argument("--out", dest="out_path", default=None, help="single output .npy path")
    p.add_argument("--no-canon", action="store_true", help="disable torso canonicalization")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)

def main(argv: list):
    args = parse_args(argv)
    use_canon = (not args.no_canon)

    # 단일 파일 모드
    if args.in_path and args.out_path:
        ok, msg = process_one(args.in_path, args.out_path, use_canon, args.overwrite)
        print(msg)
        return

    # 폴더 모드
    if not (args.keypoints_dir and args.features_dir):
        print("Either use --in/--out for single file, or --keypoints-dir/--features-dir for batch.")
        sys.exit(2)

    in_list = list_npy_recursively(args.keypoints_dir)
    if not in_list:
        print(f"No .npy under '{args.keypoints_dir}'")
        return

    print(f"Found {len(in_list)} holistic files → features to '{args.features_dir}' (canon={use_canon})")
    ok_cnt = 0
    for src in tqdm(in_list, desc="Converting"):
        rel = os.path.relpath(src, args.keypoints_dir)
        dst = os.path.join(args.features_dir, rel)
        ok, msg = process_one(src, dst, use_canon, args.overwrite)
        if ok: ok_cnt += 1
        else:  print(msg)
    print(f"Done. success={ok_cnt}/{len(in_list)}")

if __name__ == "__main__":
    main(sys.argv[1:])
