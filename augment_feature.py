#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-Features (T,152) Augmentation Pipeline — 폴더 구조 유지 + 원본 포함 복제

입력 특징 벡터(프레임마다 152D):
  [0:63]   : left hand relative coords (21*3)
  [63:126] : right hand relative coords (21*3)
  [126:136]: left hand angles (10)
  [136:146]: right hand angles (10)
  [146:152]: face(nose) -> wrists (6)

지원 증강 (조합 적용):
  Spatial(좌표 성분만): Small 3D rotation, Scale jitter, Gaussian noise
  Temporal: Time-stretch(선형보간), Time-mask(구간 0마스킹), Window drop(일부 프레임 삭제)

길이 정책:
  --mode keep  : 원래 길이 유지(증강 이후 원래 T로 재보간)
  --mode force : --target-len 길이로 강제(예: 90)

출력 규칙:
  - 폴더 모드: dst/ + (src 기준 상대경로)로 저장 (원본 1개 + *_augK.npy)
  - 단일 파일 모드: --out 경로에 원본 1개 + *_augK.npy 저장
  - 기본적으로 --save-original 활성화(원본 복제). 끄려면 --no-save-original
"""

import os
import sys
import argparse
from typing import Tuple, List, Optional

import numpy as np
from tqdm import tqdm

# ---- 고정 인덱스 (152D feature의 블럭 경계) ----
LH_BEG, LH_END = 0, 63         # 21*3
RH_BEG, RH_END = 63, 126       # 21*3
LA_BEG, LA_END = 126, 136      # 10
RA_BEG, RA_END = 136, 146      # 10
FR_BEG, FR_END = 146, 152      # 6

COORD_SLICES = [(LH_BEG, LH_END), (RH_BEG, RH_END), (FR_BEG, FR_END)]

# ---- 유틸 ----
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

def rand_uniform(a: float, b: float) -> float:
    return float(np.random.uniform(a, b))

# ---- Spatial Augs ----
def rot_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.deg2rad(rx_deg), np.deg2rad(ry_deg), np.deg2rad(rz_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float32)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float32)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float32)
    # R = Rz * Ry * Rx
    return (Rz @ Ry @ Rx).astype(np.float32)

def apply_rotation_to_coords(frame_152: np.ndarray, R: np.ndarray) -> np.ndarray:
    out = frame_152.copy()
    for beg, end in COORD_SLICES:
        blk = out[beg:end]
        if end - beg == 63:
            pts = blk.reshape(21, 3) @ R.T
            out[beg:end] = pts.reshape(-1)
        elif end - beg == 6:
            pts = blk.reshape(2, 3) @ R.T
            out[beg:end] = pts.reshape(-1)
    return out

def apply_scale_to_coords(frame_152: np.ndarray, scale: float) -> np.ndarray:
    out = frame_152.copy()
    for beg, end in COORD_SLICES:
        out[beg:end] *= scale
    return out

def apply_coord_noise(frame_152: np.ndarray, sigma: float) -> np.ndarray:
    out = frame_152.copy()
    for beg, end in COORD_SLICES:
        out[beg:end] += np.random.normal(0.0, sigma, size=(end - beg,)).astype(np.float32)
    return out

# ---- Temporal Augs ----
def resample_linear(seq: np.ndarray, new_T: int) -> np.ndarray:
    """
    seq: (T, D) → (new_T, D)
    """
    T, D = seq.shape
    if T == new_T:
        return seq.copy()
    if T == 0 or new_T <= 0:
        return np.zeros((max(new_T, 0), D), dtype=np.float32)

    xs = np.linspace(0, T - 1, num=T, dtype=np.float32)
    xs_new = np.linspace(0, T - 1, num=new_T, dtype=np.float32)

    idx0 = np.floor(xs_new).astype(np.int64)
    idx1 = np.clip(idx0 + 1, 0, T - 1)
    t = (xs_new - idx0).reshape(-1, 1)  # (new_T,1)

    s0 = seq[idx0]  # (new_T,D)
    s1 = seq[idx1]  # (new_T,D)
    out = (1 - t) * s0 + t * s1
    return out.astype(np.float32)

def time_stretch(seq: np.ndarray, min_rate: float, max_rate: float, mode: str, target_len: Optional[int]) -> np.ndarray:
    """
    시간 스트레치. mode='keep'이면 원래 T 또는 target_len 기준 유지.
    """
    T, _ = seq.shape
    rate = rand_uniform(min_rate, max_rate)
    new_T = max(int(round(T * rate)), 1)
    stretched = resample_linear(seq, new_T)

    if mode == "keep":
        return resample_linear(stretched, T)
    elif mode == "force":
        assert target_len is not None and target_len > 0
        return resample_linear(stretched, target_len)
    else:
        return stretched

def time_mask(seq: np.ndarray, p: float, max_width_ratio: float) -> np.ndarray:
    """
    확률 p로 일정 구간(폭<=max_width_ratio*T)을 0으로 마스킹
    """
    T, _ = seq.shape
    out = seq.copy()
    if np.random.rand() > p or T == 0:
        return out
    max_w = max(1, int(T * max_width_ratio))
    w = np.random.randint(1, max_w + 1)
    s = np.random.randint(0, max(1, T - w + 1))
    out[s:s + w, :] = 0.0
    return out

def window_drop(seq: np.ndarray, p: float, max_drop_ratio: float, mode: str, target_len: Optional[int]) -> np.ndarray:
    """
    확률 p로 랜덤 구간을 잘라내기. keep이면 원래 길이로 보간 복구, force면 target_len으로 보간.
    """
    T, _ = seq.shape
    if np.random.rand() > p or T <= 2:
        return seq.copy()
    max_drop = max(1, int(T * max_drop_ratio))
    drop = np.random.randint(1, max_drop + 1)
    start = np.random.randint(0, max(1, T - drop + 1))
    kept = np.concatenate([seq[:start], seq[start + drop:]], axis=0)
    if kept.shape[0] < 2:
        kept = seq.copy()
    if mode == "keep":
        return resample_linear(kept, T)
    elif mode == "force":
        assert target_len is not None and target_len > 0
        return resample_linear(kept, target_len)
    else:
        return kept

# ---- 한 시퀀스에 증강 조합 적용 ----
def augment_sequence(
    seq: np.ndarray,
    mode: str = "keep",
    target_len: Optional[int] = None,
    # Spatial ranges
    max_rot_deg: float = 12.0,
    min_scale: float = 0.95,
    max_scale: float = 1.05,
    coord_noise_sigma: float = 0.005,
    # Temporal ranges
    tstretch_min: float = 0.9,
    tstretch_max: float = 1.1,
    time_mask_p: float = 0.5,
    time_mask_max_ratio: float = 0.15,
    drop_p: float = 0.35,
    drop_max_ratio: float = 0.2,
) -> np.ndarray:
    """
    seq: (T,152)
    """
    assert seq.ndim == 2 and seq.shape[1] == 152, f"Expected (T,152), got {seq.shape}"
    out = seq.copy()

    # Temporal: stretch → mask → drop
    out = time_stretch(out, tstretch_min, tstretch_max, mode, target_len)
    out = time_mask(out, time_mask_p, time_mask_max_ratio)
    out = window_drop(out, drop_p, drop_max_ratio, mode, target_len)

    # Spatial: 프레임별 동일 파라미터 적용(시계열 일관성)
    rx = rand_uniform(-max_rot_deg, max_rot_deg)
    ry = rand_uniform(-max_rot_deg, max_rot_deg)
    rz = rand_uniform(-max_rot_deg, max_rot_deg)
    R = rot_matrix_xyz(rx, ry, rz)
    s = rand_uniform(min_scale, max_scale)

    for i in range(out.shape[0]):
        f = out[i]
        f = apply_rotation_to_coords(f, R)
        f = apply_scale_to_coords(f, s)
        f = apply_coord_noise(f, coord_noise_sigma)
        out[i] = f

    return out.astype(np.float32)

# ---- 파일/폴더 처리 ----
def save_original(out_path: str, seq: np.ndarray, overwrite: bool) -> None:
    ensure_parent(out_path)
    if (not overwrite) and os.path.exists(out_path):
        return
    np.save(out_path, seq)

def process_one(
    in_path: str,
    out_path: str,
    num_aug: int,
    mode: str,
    target_len: Optional[int],
    overwrite: bool,
    seed: Optional[int],
    save_orig: bool,
    config: dict
) -> None:
    ensure_parent(out_path)
    base = os.path.splitext(out_path)[0]
    try:
        seq = np.load(in_path)
    except Exception as e:
        print(f"[FAIL] load {in_path}: {e}")
        return

    if seq.ndim != 2 or seq.shape[1] != 152:
        print(f"[SKIP] {in_path}: expected (T,152), got {seq.shape}")
        return

    # 0) 원본 저장 (그대로 복제)
    if save_orig:
        save_original(out_path, seq, overwrite)

    # 1) 증강 저장
    for k in range(num_aug):
        if seed is not None:
            np.random.seed(seed + k)
        aug = augment_sequence(
            seq,
            mode=mode,
            target_len=target_len,
            **config
        )
        save_path = f"{base}_aug{k+1}.npy"
        if (not overwrite) and os.path.exists(save_path):
            continue
        np.save(save_path, aug)

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3-Features (T,152) augmentation — keep folder tree & copy originals")
    # IO
    p.add_argument("--src", default=None, help="source folder (recursive). Use with --dst")
    p.add_argument("--dst", default=None, help="destination folder (mirror structure)")
    p.add_argument("--in", dest="in_path", default=None, help="single input .npy path")
    p.add_argument("--out", dest="out_path", default=None, help="single output .npy path (base for *_augK.npy)")
    p.add_argument("--num-aug", type=int, default=2, help="number of augmented samples per file")
    p.add_argument("--overwrite", action="store_true", help="overwrite outputs")

    # Length policy
    p.add_argument("--mode", choices=["keep", "force"], default="keep", help="'keep' keeps length; 'force' resamples to --target-len")
    p.add_argument("--target-len", type=int, default=90, help="target length if mode='force'")

    # Spatial params
    p.add_argument("--max-rot-deg", type=float, default=12.0)
    p.add_argument("--min-scale", type=float, default=0.95)
    p.add_argument("--max-scale", type=float, default=1.05)
    p.add_argument("--coord-noise-sigma", type=float, default=0.005)

    # Temporal params
    p.add_argument("--tstretch-min", type=float, default=0.9)
    p.add_argument("--tstretch-max", type=float, default=1.1)
    p.add_argument("--time-mask-p", type=float, default=0.5)
    p.add_argument("--time-mask-max-ratio", type=float, default=0.15)
    p.add_argument("--drop-p", type=float, default=0.35)
    p.add_argument("--drop-max-ratio", type=float, default=0.2)

    p.add_argument("--seed", type=int, default=None)

    # Save original
    p.add_argument("--save-original", dest="save_original", action="store_true", default=True,
                   help="save original sequence alongside augmented ones (default: on)")
    p.add_argument("--no-save-original", dest="save_original", action="store_false",
                   help="do not save the original copy")
    return p.parse_args(argv)

def main(argv: List[str]) -> None:
    args = parse_args(argv)

    config = dict(
        max_rot_deg=args.max_rot_deg,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        coord_noise_sigma=args.coord_noise_sigma,
        tstretch_min=args.tstretch_min,
        tstretch_max=args.tstretch_max,
        time_mask_p=args.time_mask_p,
        time_mask_max_ratio=args.time_mask_max_ratio,
        drop_p=args.drop_p,
        drop_max_ratio=args.drop_max_ratio,
    )

    # 단일 파일 모드
    if args.in_path and args.out_path:
        process_one(
            args.in_path,
            args.out_path,
            num_aug=args.num_aug,
            mode=args.mode,
            target_len=(args.target_len if args.mode == "force" else None),
            overwrite=args.overwrite,
            seed=args.seed,
            save_orig=args.save_original,
            config=config,
        )
        print(f"[DONE] single: {args.in_path} -> {args.out_path} (+ *_aug*.npy)")
        return

    # 폴더 모드
    if not (args.src and args.dst):
        print("Use --in/--out for single file, or --src/--dst for batch processing.")
        sys.exit(2)

    files = list_npy_recursively(args.src)
    if not files:
        print(f"No .npy files under '{args.src}'")
        return

    print(f"Found {len(files)} files. save_original={args.save_original}, mode={args.mode}, "
          f"target_len={args.target_len if args.mode=='force' else 'N/A'}")

    done = 0
    for src in tqdm(files, desc="Augmenting"):
        rel = os.path.relpath(src, args.src)
        dst_path = os.path.join(args.dst, rel)  # 동일 파일명으로 원본/증강 저장
        process_one(
            src, dst_path,
            num_aug=args.num_aug,
            mode=args.mode,
            target_len=(args.target_len if args.mode == "force" else None),
            overwrite=args.overwrite,
            seed=args.seed,
            save_orig=args.save_original,
            config=config,
        )
        done += 1
    print(f"[DONE] processed={done}, out='{args.dst}'")

if __name__ == "__main__":
    main(sys.argv[1:])
