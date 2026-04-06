#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
multi_meld_mlp.py — Generation 2 network: hand + last 4 plays → action logits.

Input encoding (270 bits = 5 × 54):
    hand          : 54 bits — cumulative count per value (StateEncoder._encode_hand)
    meld_right    : 54 bits — last play by player 1 step counter-clockwise (current target)
    meld_opposite : 54 bits — last play by player 2 steps counter-clockwise
    meld_left     : 54 bits — last play by player 3 steps counter-clockwise
    meld_self     : 54 bits — player's own last play (4 steps counter-clockwise)

Each meld slot uses StateEncoder.encode_meld(), zeros if no play yet (opening lead / pass).

Output: 55-class logits (0-53 melds, 54 = pass)

encode_state() is the single source of truth for this generation's input format.
Any alternative architecture over the same input (e.g. MultiMeldTransformer)
should import and reuse encode_state() from this module rather than redefining it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from president.core.Meld import Meld
from president.training.data.StateEncoder import StateEncoder, MELD_BITS

ACTIVATIONS = {
    "relu":      nn.ReLU,
    "tanh":      nn.Tanh,
    "leakyrelu": nn.LeakyReLU,
}

INPUT_SIZE  = 270   # 54 hand + 4 × 54 meld history
OUTPUT_SIZE = 55    # 0-53 melds, 54 = pass

# Number of meld context slots (right, opposite, left, self)
NUM_MELDS = 4


# ─────────────────────────────────────────────
# Input encoding — shared by all generation-2 architectures
# ─────────────────────────────────────────────

def encode_state(
    hand: list,
    melds: list,
) -> np.ndarray:
    """
    Encode (hand, [meld_right, meld_opposite, meld_left, meld_self]) into the
    270-bit input vector for this generation.

    Args:
        hand:  list of PlayingCard
        melds: list of 4 Optional[Meld] — [meld_right, meld_opposite, meld_left, meld_self]
               zeros are used for any slot that is None

    Returns:
        np.ndarray of shape (270,), dtype int8
    """
    if len(melds) != NUM_MELDS:
        raise ValueError(f"encode_state expects {NUM_MELDS} melds, got {len(melds)}")

    hand_enc = StateEncoder._encode_hand(hand)
    meld_encs = [
        StateEncoder.encode_meld(m) if m is not None else np.zeros(MELD_BITS, dtype=np.int8)
        for m in melds
    ]
    return np.concatenate([hand_enc] + meld_encs)


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class MultiMeldMLP(nn.Module):
    """
    MLP that maps (hand, last-4-plays) to action logits.

    Architecture is fully config-driven via constructor args.
    Use encode_state() to build inputs for both training and inference.

    The MULTI_MELD class attribute signals to the data generator that this
    model expects a list of NUM_MELDS meld slots rather than a single target.
    """

    INPUT_SIZE  = INPUT_SIZE
    OUTPUT_SIZE = OUTPUT_SIZE
    MULTI_MELD  = True   # data-generator feature flag

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
            x: (batch, 270) float tensor
        Returns:
            logits: (batch, 55) float tensor
        """
        return self.net(x)

    @staticmethod
    def encode_state(hand: list, melds: list) -> np.ndarray:
        """Convenience alias — delegates to module-level encode_state()."""
        return encode_state(hand, melds)
