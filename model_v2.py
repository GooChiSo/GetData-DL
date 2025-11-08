#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
키포인트 기반 GRU 분류 모델(v2, 마스크 대응 어텐션)

입력:
  x: (B, T, 152)
  lengths: (B,)  실제 길이(패딩 제외)
어텐션:
  - 입력 x의 앞쪽 146차원을 hand_feat로 사용(좌표+각도)
  - padding 위치는 어텐션에서 자동 마스킹
출력:
  logits: (B, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class KeypointGRUModelV2(nn.Module):
    def __init__(self, input_dim=152, attn_dim=146, hidden_dim=256, num_classes=6, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        enc_dim = hidden_dim * (2 if bidirectional else 1)

        self.attn_proj = nn.Sequential(
            nn.Linear(attn_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)  # (B,T,1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(enc_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, lengths):
        """
        x: (B,T,152), lengths: (B,)
        """
        B, T, D = x.shape
        assert D == self.input_dim, f"Expected last dim {self.input_dim}, got {D}"

        # GRU 인코딩 (패킹은 생략하고, 마스크로 가중합 처리)
        rnn_out, _ = self.gru(x)  # (B,T,H)

        # 어텐션(마스크 적용)
        hand_feat = x[:, :, :self.attn_dim]  # (B,T,146)
        attn_logits = self.attn_proj(hand_feat).squeeze(-1)  # (B,T)

        # lengths → mask (B,T), 1: valid, 0: pad
        device = x.device
        idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B,T)
        mask = (idxs < lengths.unsqueeze(1)).to(torch.bool)              # (B,T)

        # 패딩 위치는 매우 작은 값으로 마스킹
        minus_inf = torch.finfo(attn_logits.dtype).min
        attn_logits_masked = attn_logits.masked_fill(~mask, minus_inf)
        attn_weights = F.softmax(attn_logits_masked, dim=1).unsqueeze(-1)  # (B,T,1)

        # 마스크된 가중합 (패딩 위치의 가중치는 0이 됨)
        feat = (rnn_out * attn_weights).sum(dim=1)  # (B,H)

        return self.classifier(feat)
