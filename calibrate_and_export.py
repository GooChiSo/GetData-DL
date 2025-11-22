#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration exporter: Temperature scaling (T), exponential squashing (beta), 
and per-class precision-based 5-level thresholds (tau_1..tau_4) + margins.

가정(입력 데이터 구조 중 하나를 선택):
A) --val-dir 사용 시: 폴더 구조  val_dir/<label>/*.npy   각 npy는 (T,152)
B) --val-csv 사용 시: CSV의 두 컬럼 path,label  (path는 npy 경로)
- 두 옵션 중 하나만 쓰면 됨.

출력:
  model/<CATEGORY>_calib.json  (T, beta, thresholds, margins, meta)
"""

import os, json, argparse, glob, csv
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle

from model_v2 import KeypointGRUModelV2  # 네 프로젝트의 같은 경로에 있다고 가정

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 유틸
# -----------------------------
def load_label_map(path):
    with open(path, "rb") as f:
        lm = pickle.load(f)
    idx_to_label = {v:k for k,v in lm.items()}
    return lm, idx_to_label

def iter_npy_from_dir(val_dir, label_map):
    for lab in os.listdir(val_dir):
        lab_dir = os.path.join(val_dir, lab)
        if not os.path.isdir(lab_dir): 
            continue
        if lab not in label_map:
            # 라벨맵에 없는 라벨은 스킵
            continue
        for p in glob.glob(os.path.join(lab_dir, "*.npy")):
            yield p, lab

def iter_npy_from_csv(val_csv):
    with open(val_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row["path"], row["label"]

def padpack_tensor(xs):
    """xs: (T,152) numpy -> (1,T,152) tensor + length"""
    T = xs.shape[0]
    x = torch.tensor(xs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    L = torch.tensor([T], dtype=torch.long, device=DEVICE)
    return x, L

def softmax_with_temperature(logits, T):
    return F.softmax(logits / T, dim=-1)

def exp_squash(p, beta):
    # p in [0,1], beta<0 권장
    # f(p) = (exp(beta*p)-1)/(exp(beta)-1)
    eb = np.exp(beta)
    num = np.exp(beta * p) - 1.0
    den = eb - 1.0 + 1e-12
    return (num / den).clip(0.0, 1.0)

# ECE (binary-style using top-1 confidence), bins=15
def expected_calibration_error(conf, correct, n_bins=15):
    conf = np.asarray(conf)
    correct = np.asarray(correct).astype(np.float32)
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (conf >= lo) & (conf < hi) if i < n_bins-1 else (conf >= lo) & (conf <= hi)
        if not np.any(m): 
            continue
        acc = correct[m].mean()
        avg_conf = conf[m].mean()
        ece += (m.sum()/n) * abs(avg_conf - acc)
    return float(ece)

def compute_precision_thresholds_per_class(records, idx_to_label, targets=(0.50, 0.70, 0.85, 0.95)):
    """
    records: list of dict per sample
      { 'y_true':int, 'y_pred':int, 'p_prime':float, 'top2_p_prime':float }
    클래스별로 임계값을 Precision >= target 이 되는 최소 임계로 잡음.
    반환: {label: {'tau': [t1,t2,t3,t4], 'stats': {...}}}
    """
    C = len(idx_to_label)
    # 클래스별 레코드 분리 (top1==c 인 샘플만 임계 평가의 母집단)
    per_class = {c: [] for c in range(C)}
    for r in records:
        per_class[r['y_pred']].append(r)

    out = {}
    for c in range(C):
        lab = idx_to_label[c]
        rc = per_class[c]
        if len(rc) == 0:
            # 예측이 한 번도 안된 클래스: 극단값으로 막음
            out[lab] = {'tau':[1.01,1.01,1.01,1.01], 'stats': {'count':0}}
            continue
        # p' 값 후보
        ps = np.array([r['p_prime'] for r in rc], dtype=np.float32)
        ys = np.array([1 if r['y_true']==c else 0 for r in rc], dtype=np.int32)

        # 후보 임계: 유니크 p' (조밀하지 않으면 linspace로 보강)
        unique_ps = np.unique(ps)
        grid = np.sort(unique_ps)
        if len(grid) < 50:
            grid = np.unique(np.concatenate([grid, np.linspace(0,1,101)]))
        tau = []
        for tar in targets:
            found = 1.01  # 기본 불가능 (disable)
            for t in grid:
                m = ps >= t
                if not np.any(m):
                    continue
                prec = ys[m].sum() / float(m.sum())
                if prec >= tar:
                    found = float(t)
                    break
            tau.append(found)
        out[lab] = {'tau': tau, 'stats': {'count': int(len(rc))}}
    return out

# -----------------------------
# 메인
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, help="카테고리명 (모델/라벨맵 파일 접두)")
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--val-dir", default=None, help="A) 폴더 구조 val_dir/<label>/*.npy")
    ap.add_argument("--val-csv", default=None, help="B) CSV with columns: path,label")
    ap.add_argument("--attn-dim", type=int, default=146)
    ap.add_argument("--input-dim", type=int, default=152)
    ap.add_argument("--bins", type=int, default=15, help="ECE bins")
    ap.add_argument("--beta-min", type=float, default=-6.0)
    ap.add_argument("--beta-max", type=float, default=-0.5)
    ap.add_argument("--beta-steps", type=int, default=24)
    ap.add_argument("--margins", default="0,0.02,0.05,0.08,0.12",
                    help="m1..m5 (comma) for L1..L5 min margins")
    args = ap.parse_args()

    # 경로
    label_map_path = os.path.join(args.model_dir, f"{args.category}_label_map.pkl")
    model_path     = os.path.join(args.model_dir, f"{args.category}_model.pth")
    calib_out      = os.path.join(args.model_dir, f"{args.category}_calib.json")

    # 로드
    label_map, idx_to_label = load_label_map(label_map_path)
    num_classes = len(label_map)
    model = KeypointGRUModelV2(input_dim=args.input_dim, attn_dim=args.attn_dim, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 데이터 소스
    samples = []
    if (args.val_dir is None) ^ (args.val_csv is None) == False:
        print("[ERR] --val-dir 또는 --val-csv 중 정확히 하나만 지정하세요.")
        return
    if args.val_dir:
        for path, lab in iter_npy_from_dir(args.val_dir, label_map):
            samples.append((path, lab))
    else:
        for path, lab in iter_npy_from_csv(args.val_csv):
            samples.append((path, lab))
    if len(samples) == 0:
        print("[ERR] 보정용 샘플이 없습니다.")
        return

    # 1) Temperature scaling - 간단히 로그-탐색으로 근사 (NLL 최소화)
    #   더 정밀하게 하려면 LBFGS로 미세최적화 가능.
    T_grid = np.linspace(0.5, 3.0, 26)  # 0.5~3.0
    best_T, best_nll = 1.0, float("inf")

    # 먼저 logits/labels를 한 번 수집
    logit_list, ytrue_list = [], []
    with torch.no_grad():
        for path, lab in tqdm(samples, desc="Collect logits"):
            x = np.load(path)  # (T,152)
            xb, L = padpack_tensor(x)
            logits = model(xb, L)  # [1,C]
            logit_list.append(logits.squeeze(0).cpu().numpy())
            ytrue_list.append(label_map[lab])
    logits_all = np.stack(logit_list, axis=0)  # [N,C]
    ytrue_all = np.array(ytrue_list, dtype=np.int64)

    for T in T_grid:
        q = softmax_with_temperature(torch.tensor(logits_all), T=T)
        # NLL
        nll = F.nll_loss(torch.log(q+1e-12), torch.tensor(ytrue_all)).item()
        if nll < best_nll:
            best_nll, best_T = nll, T

    # 2) Beta grid (ECE 최소)
    beta_grid = np.linspace(args.beta_min, args.beta_max, args.beta_steps)
    best_beta, best_ece = -2.0, float("inf")

    with torch.no_grad():
        qT = softmax_with_temperature(torch.tensor(logits_all), T=best_T).cpu().numpy()
        top1 = qT.argmax(axis=1)
        top1_p = qT[np.arange(len(qT)), top1]
        correct = (top1 == ytrue_all).astype(np.int32)

    for beta in beta_grid:
        pprime = exp_squash(top1_p, beta=beta)
        ece = expected_calibration_error(pprime, correct, n_bins=args.bins)
        if ece < best_ece:
            best_ece, best_beta = ece, beta

    # 3) 클래스별 임계 (정밀도 기준)
    # 보정 확률과 second top도 뽑아서 기록
    with torch.no_grad():
        qT = softmax_with_temperature(torch.tensor(logits_all), T=best_T).cpu().numpy()
    # 두 번째 확률
    part_sorted = np.sort(qT, axis=1)
    top2_p = part_sorted[:, -2]
    # 보정 확률 p'
    pprime = exp_squash(qT.max(axis=1), beta=best_beta)
    top2_pprime = exp_squash(top2_p, beta=best_beta)

    records = []
    for i in range(len(qT)):
        records.append({
            'y_true': int(ytrue_all[i]),
            'y_pred': int(np.argmax(qT[i])),
            'p_prime': float(pprime[i]),
            'top2_p_prime': float(top2_pprime[i]),
        })
    thresholds = compute_precision_thresholds_per_class(records, idx_to_label)

    # 4) 마진 파라미터
    margins = [float(x) for x in args.margins.split(",")]
    if len(margins) != 5:
        raise ValueError("--margins 는 5개 값을 콤마로 주세요 (예: 0,0.02,0.05,0.08,0.12)")

    # 5) 저장
    payload = {
        "category": args.category,
        "temperature": float(best_T),
        "beta": float(best_beta),
        "ece": float(best_ece),
        "thresholds": thresholds,   # {label: {'tau':[...], 'stats':{...}}}
        "margins": {
            "L1": margins[0],
            "L2": margins[1],
            "L3": margins[2],
            "L4": margins[3],
            "L5": margins[4],
        },
        "meta": {
            "bins": int(args.bins),
            "targets": [0.50, 0.70, 0.85, 0.95],
            "note": "tau_k는 Precision >= target 을 만족하는 최소 p' 임계값"
        }
    }
    os.makedirs(args.model_dir, exist_ok=True)
    with open(calib_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved calibration → {calib_out}")
    print(f"     T={payload['temperature']:.3f}, beta={payload['beta']:.3f}, ECE={payload['ece']:.4f}")
    print("     Example thresholds of a class:", next(iter(thresholds.items())))
    return

if __name__ == "__main__":
    main()
