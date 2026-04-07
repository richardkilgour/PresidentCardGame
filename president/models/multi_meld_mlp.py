#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
multi_meld_mlp.py — Generation 2 network: hand + last 4 actions → action logits.

Input encoding (282 bits = 54 + 4 × 57):
    hand : 54 bits — cumulative count per value (StateEncoder._encode_hand)
    t-1  : 57 bits — action 1 step back (counterclockwise from current player)
    t-2  : 57 bits — action 2 steps back
    t-3  : 57 bits — action 3 steps back
    t-4  : 57 bits — current player's own last action

Each slot is one-hot (57 bits):
    bits 0–53 : valid meld (StateEncoder.encode_meld)
    bit 54    : pass
    bit 55    : finished
    bit 56    : waiting (also used when history is exhausted)

ROUND_WON markers are skipped; only player actions count as time steps.
If fewer than 4 actions exist in history, remaining slots are encoded as waiting.

Output: 55-class logits (0-53 melds, 54 = pass)

MultiMeldMLP.encode_state() is the single source of truth for this generation's
input format. Any alternative architecture over the same input (e.g. MultiMeldTransformer)
should import and reuse encode_state() from this module rather than redefining it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from president.core.Meld import Meld
from president.core.PlayHistory import PlayHistory, EventType
from president.training.data.StateEncoder import StateEncoder, MELD_BITS

ACTIVATIONS = {
    "relu":      nn.ReLU,
    "tanh":      nn.Tanh,
    "leakyrelu": nn.LeakyReLU,
}

NUM_MELDS    = 4
SLOT_BITS    = MELD_BITS + 3        # 57: meld(54) | pass(1) | finished(1) | waiting(1)
IDX_PASS     = MELD_BITS            # 54
IDX_FINISHED = MELD_BITS + 1        # 55
IDX_WAITING  = MELD_BITS + 2        # 56

INPUT_SIZE  = 54 + NUM_MELDS * SLOT_BITS   # 54 hand + 4 × 57 = 282
OUTPUT_SIZE = 55                           # 0-53 melds, 54 = pass


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
            x: (batch, 282) float tensor
        Returns:
            logits: (batch, 55) float tensor
        """
        return self.net(x)

    @staticmethod
    def encode_state(play_history: PlayHistory, player) -> np.ndarray:
        """
        Encode state by stepping back through the last 4 player actions.

        Steps counterclockwise through time (t-1, t-2, t-3, t-4) using
        previous_plays_generator. ROUND_WON markers are skipped.

        Each slot is one-hot over 57 bits:
            bits 0–53 : valid meld
            bit 54    : pass
            bit 55    : finished
            bit 56    : waiting (also used when history is exhausted)
        """
        slots = []
        for _, meld_or_code, _ in play_history.previous_plays_generator():
            if meld_or_code == -1:          # ROUND_WON — not a player action
                continue
            slot = np.zeros(SLOT_BITS, dtype=np.int8)
            if isinstance(meld_or_code, Meld):
                if meld_or_code.cards:
                    slot[:MELD_BITS] = StateEncoder.encode_meld(meld_or_code)
                else:
                    slot[IDX_PASS] = 1
            elif isinstance(meld_or_code, int):  # COMPLETE (rank index)
                slot[IDX_FINISHED] = 1
            else:                               # None — WAITING
                slot[IDX_WAITING] = 1
            slots.append(slot)
            if len(slots) == NUM_MELDS:
                break

        while len(slots) < NUM_MELDS:
            slot = np.zeros(SLOT_BITS, dtype=np.int8)
            slot[IDX_WAITING] = 1
            slots.append(slot)

        hand_enc = StateEncoder._encode_hand(player._hand)
        return np.concatenate([hand_enc] + slots)
