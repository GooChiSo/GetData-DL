#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numpy 시퀀스(.npy) 로더용 Dataset

폴더 구조(카테고리 단위):
  data_root/
    ├─ 카테고리A/
    │    ├─ 단어1/*.npy
    │    ├─ 단어2/*.npy
    │    └─ ...
    └─ 카테고리B/ ...

옵션:
  - include_orig: 원본(.npy, *_aug가 아닌 파일) 포함 여부
  - include_aug : 증강(*_aug*.npy) 포함 여부
필요 시 class_filter로 특정 단어만 학습 가능
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

def is_aug_file(filename: str) -> bool:
    fname = os.path.basename(filename)
    name, ext = os.path.splitext(fname)
    return ("_aug" in name) and (ext.lower() == ".npy")

class NpySequenceDataset(Dataset):
    def __init__(self, category_dir: str, label_map: dict,
                 include_orig: bool = True,
                 include_aug: bool = True,
                 require_feature_dim: int = 152,
                 class_filter: list | None = None):
        """
        category_dir: data_root/카테고리
        label_map: {단어명: 정수라벨}
        """
        self.samples: list[tuple[str, int]] = []
        self.label_map = label_map
        self.require_feature_dim = require_feature_dim

        for word in sorted(os.listdir(category_dir)):
            word_path = os.path.join(category_dir, word)
            if not os.path.isdir(word_path):
                continue
            if class_filter is not None and word not in class_filter:
                continue
            label = label_map.get(word)
            if label is None:
                continue

            for file in sorted(os.listdir(word_path)):
                if not file.lower().endswith(".npy"):
                    continue
                full = os.path.join(word_path, file)
                is_aug = is_aug_file(full)
                if (not include_aug) and is_aug:
                    continue
                if (not include_orig) and (not is_aug):
                    continue
                self.samples.append((full, label))

        if len(self.samples) == 0:
            print(f"[WARN] no samples under: {category_dir} (orig={include_orig}, aug={include_aug})")
        else:
            print(f"[INFO] {category_dir}: loaded {len(self.samples)} samples (orig={include_orig}, aug={include_aug})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = np.load(path)  # (T, F)
        if x.ndim != 2 or x.shape[1] != self.require_feature_dim:
            raise ValueError(f"Bad shape {x.shape} at {path}; expected (T,{self.require_feature_dim})")
        x = torch.tensor(x, dtype=torch.float32)   # (T, F)
        y = torch.tensor(label, dtype=torch.long)  # ()
        return x, y
