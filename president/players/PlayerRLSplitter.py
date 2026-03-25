#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlayerRLSplitter — an AbstractPlayer that uses the RL-trained MaskablePPO
policy to play President.

Loads the best model saved by EvalCallback during RL training.
Falls back to the first legal play if the model predicts an illegal action.
Uses direct policy network inference for speed — bypasses SB3 overhead.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from president.core.AbstractPlayer import AbstractPlayer
from president.core.Meld import Meld
from president.core.StateEncoder import StateEncoder

MODEL_PATH  = Path(__file__).parent.parent / "models" / "best_model.zip"
MELD_BITS   = 54
ACTION_BITS = 55

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlayerRLSplitter(AbstractPlayer):
    """Plays using the RL-trained MaskablePPO policy."""

    _device = _device
    _model  = MaskablePPO.load(MODEL_PATH, device=_device)

    def play(self) -> Meld:
        obs  = self._get_observation()
        mask = self._get_action_mask()

        with torch.no_grad():
            obs_tensor  = torch.tensor(obs,  dtype=torch.float32).unsqueeze(0).to(self._device)
            mask_tensor = torch.tensor(mask, dtype=torch.bool   ).unsqueeze(0).to(self._device)

            # Get logits directly from the policy network
            distribution = self._model.policy.get_distribution(obs_tensor)
            logits       = distribution.distribution.logits.clone()

            # Mask illegal actions
            logits[~mask_tensor] = float("-inf")
            action = logits.argmax(dim=1).item()

        meld  = self._action_to_meld(action)
        legal = self.possible_plays()

        if meld in legal:
            return meld

        logging.warning(
            f"{self.name}: RL model predicted illegal meld, falling back. "
            f"Hand: {self._hand}, target: {self.target_meld}"
        )
        return legal[0]

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        hand_enc   = StateEncoder._encode_hand(self._hand)
        target_enc = StateEncoder.encode_meld(self.target_meld) \
                     if self.target_meld is not None \
                     else np.zeros(MELD_BITS, dtype=np.float32)
        return np.concatenate([hand_enc, target_enc]).astype(np.float32)

    def _get_action_mask(self) -> np.ndarray:
        mask  = np.zeros(ACTION_BITS, dtype=bool)
        legal = self.possible_plays()
        for meld in legal:
            mask[self._meld_to_action(meld)] = True
        return mask

    def _action_to_meld(self, action: int) -> Meld:
        if action == 54:
            return Meld()
        value = action // 4
        count = action % 4 + 1
        candidates = [c for c in self._hand if c.get_value() == value]
        if len(candidates) < count:
            return Meld()
        meld = None
        for card in candidates[:count]:
            meld = Meld(card, meld) if meld else Meld(card)
        return meld

    @staticmethod
    def _meld_to_action(meld: Meld) -> int:
        if not meld.cards:
            return 54
        value = meld.cards[0].get_value()
        count = len(meld.cards)
        return value * 4 + count - 1
