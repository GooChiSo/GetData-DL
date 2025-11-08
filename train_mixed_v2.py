#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
카테고리별 혼합 데이터로 GRU(v2, 어텐션+마스크) 모델 학습

- data_root/카테고리/단어/*.npy  (같은 폴더에 원본+증강 공존)
- 가변 길이 T 지원 (pad + lengths mask)
- 카테고리별로 모델/.pth & 라벨맵/.pkl 저장
"""

import os
import argparse
import pickle
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from npy_sequence_dataset import NpySequenceDataset
from model_v2 import KeypointGRUModelV2

# -------- Collate: variable length pad --------
def collate_pad(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_value: float = 0.0):
    """
    batch: list of (x[T,F], y)
    returns:
      x_pad: (B, T_max, F)
      lengths: (B,)
      y: (B,)
    """
    xs, ys = zip(*batch)
    lengths = torch.tensor([t.shape[0] for t in xs], dtype=torch.long)
    T_max = int(lengths.max().item())
    F = xs[0].shape[1]
    B = len(xs)

    x_pad = torch.full((B, T_max, F), pad_value, dtype=torch.float32)
    for i, x in enumerate(xs):
        t = x.shape[0]
        x_pad[i, :t, :] = x
    y = torch.stack(ys, dim=0)
    return x_pad, lengths, y

def train_one_category(
    data_root: str,
    category: str,
    model_save_dir: str,
    include_orig: bool = True,
    include_aug: bool = True,
    input_dim: int = 152,
    attn_dim: int = 146,
    hidden_dim: int = 256,
    num_layers: int = 1,
    bidirectional: bool = False,
    batch_size: int = 8,
    epochs: int = 20,
    lr: float = 1e-4,
    val_ratio: float = 0.1,
    seed: int = 42,
    device: torch.device | None = None,
):
    torch.manual_seed(seed)

    category_dir = os.path.join(data_root, category)
    if not os.path.isdir(category_dir):
        print(f"[SKIP] not a category dir: {category_dir}")
        return

    # 1) 라벨맵: 카테고리/ 하위의 '단어' 폴더명 기준
    words = sorted([w for w in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, w))])
    if len(words) == 0:
        print(f"[SKIP] empty category: {category_dir}")
        return
    label_map = {w: i for i, w in enumerate(words)}
    print(f"[CAT:{category}] classes={len(label_map)}")

    # 2) 데이터셋
    dataset = NpySequenceDataset(
        category_dir=category_dir,
        label_map=label_map,
        include_orig=include_orig,
        include_aug=include_aug,
        require_feature_dim=input_dim,
    )
    if len(dataset) == 0:
        print(f"[SKIP] no samples: {category_dir}")
        return

    # 3) train/val split
    val_sz = max(1, int(round(len(dataset) * val_ratio))) if len(dataset) > 10 else max(1, len(dataset)//10 or 1)
    train_sz = len(dataset) - val_sz
    train_set, val_set = random_split(dataset, [train_sz, val_sz], generator=torch.Generator().manual_seed(seed))
    print(f"[CAT:{category}] train={len(train_set)}, val={len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_pad, num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_pad, num_workers=0)

    # 4) 모델/옵티마이저
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KeypointGRUModelV2(
        input_dim=input_dim, attn_dim=attn_dim,
        hidden_dim=hidden_dim, num_layers=num_layers,
        bidirectional=bidirectional, num_classes=len(label_map)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 5) 학습 루프
    best_val_acc = -1.0
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"{category}_model.pth")
    label_path = os.path.join(model_save_dir, f"{category}_label_map.pkl")

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tr_loss, tr_total, tr_correct = 0.0, 0, 0
        for x_pad, lengths, y in train_loader:
            x_pad = x_pad.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x_pad, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            tr_total += y.size(0)
            tr_correct += (preds == y).sum().item()

        tr_loss /= max(1, tr_total)
        tr_acc = 100.0 * tr_correct / max(1, tr_total)

        # ---- val ----
        model.eval()
        va_loss, va_total, va_correct = 0.0, 0, 0
        with torch.no_grad():
            for x_pad, lengths, y in val_loader:
                x_pad = x_pad.to(device)
                lengths = lengths.to(device)
                y = y.to(device)

                logits = model(x_pad, lengths)
                loss = criterion(logits, y)

                va_loss += loss.item() * y.size(0)
                preds = logits.argmax(dim=1)
                va_total += y.size(0)
                va_correct += (preds == y).sum().item()

        va_loss /= max(1, va_total)
        va_acc = 100.0 * va_correct / max(1, va_total)

        print(f"[{category}] Epoch {epoch:02d}/{epochs} | "
              f"Train Loss {tr_loss:.4f} Acc {tr_acc:.2f}% | "
              f"Val Loss {va_loss:.4f} Acc {va_acc:.2f}%")

        # best 저장
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), model_path)
            with open(label_path, "wb") as f:
                pickle.dump(label_map, f)
            print(f"  ↳ BEST updated ({best_val_acc:.2f}%) → saved: {model_path}")

    print(f"[DONE] {category} best Val Acc: {best_val_acc:.2f}% | model: {model_path}")

def parse_args():
    ap = argparse.ArgumentParser(description="Train GRU(v2)+Attention per category from feature_data_aug포함")
    ap.add_argument("--data-root", required=True, help="루트: feature_data_aug포함 (카테고리/단어/*.npy)")
    ap.add_argument("--categories", nargs="*", default=None, help="학습할 카테고리 목록(미지정 시 루트 하위 전체 폴더)")
    ap.add_argument("--model-save-dir", default="models_by_category_mixed_v2")

    ap.add_argument("--include-orig", dest="include_orig", action="store_true", default=True)
    ap.add_argument("--no-include-orig", dest="include_orig", action="store_false")
    ap.add_argument("--include-aug", dest="include_aug", action="store_true", default=True)
    ap.add_argument("--no-include-aug", dest="include_aug", action="store_false")

    ap.add_argument("--input-dim", type=int, default=152)
    ap.add_argument("--attn-dim", type=int, default=146)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--bidirectional", action="store_true", default=False)

    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()

    # 카테고리 목록 자동 수집
    if args.categories is None:
        args.categories = sorted(
            [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
        )
    print(f"[INFO] categories: {args.categories}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    for cat in args.categories:
        train_one_category(
            data_root=args.data_root,
            category=cat,
            model_save_dir=args.model_save_dir,
            include_orig=args.include_orig,
            include_aug=args.include_aug,
            input_dim=args.input_dim,
            attn_dim=args.attn_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            val_ratio=args.val_ratio,
            seed=args.seed,
            device=device,
        )

if __name__ == "__main__":
    main()
