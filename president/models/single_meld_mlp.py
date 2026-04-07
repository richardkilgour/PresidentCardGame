#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
single_meld_mlp.py — Generation 1 network: hand + target meld → action logits.

Input encoding (108 bits):
    hand   : 54 bits — cumulative count per value (StateEncoder._encode_hand)
    target : 54 bits — one-hot meld (StateEncoder.encode_meld), zeros for opening lead

Output: 55-class logits (0-53 melds, 54 = pass)

encode_state() is the single source of truth for this generation's input format.
Any alternative architecture over the same input (e.g. SingleMeldTransformer)
should import and reuse encode_state() from this module rather than redefining it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from president.core.Meld import Meld
from president.core.PlayHistory import PlayHistory
from president.training.data.StateEncoder import StateEncoder, MELD_BITS

ACTIVATIONS = {
    "relu":      nn.ReLU,
    "tanh":      nn.Tanh,
    "leakyrelu": nn.LeakyReLU,
}

INPUT_SIZE  = 108   # 54 hand + 54 target meld
OUTPUT_SIZE = 55    # 0-53 melds, 54 = pass


# ─────────────────────────────────────────────
# Input encoding — shared by all generation-1 architectures
# ─────────────────────────────────────────────

def encode_state(hand: list, target: Optional[Meld]) -> np.ndarray:
    """
    Encode (hand, target) into the 108-bit input vector for this generation.

    Args:
        hand:   list of PlayingCard
        target: current target Meld, or None for opening lead

    Returns:
        np.ndarray of shape (108,), dtype int8
    """
    hand_enc   = StateEncoder._encode_hand(hand)
    target_enc = StateEncoder.encode_meld(target) if target is not None \
                 else np.zeros(MELD_BITS, dtype=np.int8)
    return np.concatenate([hand_enc, target_enc])


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class SingleMeldMLP(nn.Module):
    """
    MLP that maps a (hand, target meld) encoding to action logits.

    Architecture is fully config-driven via constructor args.
    Use encode_state() to build inputs for both training and inference.
    """

    INPUT_SIZE  = INPUT_SIZE
    OUTPUT_SIZE = OUTPUT_SIZE

    def __init__(
        self,
        hidden_sizes: list[int] = None,
        activation:   str       = "relu",
        dropout:      float     = 0.0,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        act_cls = ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from: {list(ACTIVATIONS)}"
            )

        layers: list[nn.Module] = []
        in_size = self.INPUT_SIZE
        for hidden in hidden_sizes:
            layers += [nn.Linear(in_size, hidden), act_cls()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_size = hidden
        layers.append(nn.Linear(in_size, self.OUTPUT_SIZE))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 108) float tensor
        Returns:
            logits: (batch, 55) float tensor
        """
        return self.net(x)

    @staticmethod
    def encode_state(play_history: PlayHistory, player) -> np.ndarray:
        """Encode state from a PlayHistory and the acting player."""
        return encode_state(player._hand, play_history.current_target())
