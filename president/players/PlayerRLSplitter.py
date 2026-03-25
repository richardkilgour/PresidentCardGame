#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlayerRLSplitter — an AbstractPlayer that uses the RL-trained MaskablePPO
policy to play President.

Loads the best model saved by EvalCallback during RL training.
Falls back to the first legal play if the model predicts an illegal action.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from president.core.AbstractPlayer import AbstractPlayer
from president.core.Meld import Meld
from president.core.StateEncoder import StateEncoder

MODEL_PATH  = Path(__file__).parent.parent / "models" / "best_model.zip"
MELD_BITS   = 54
ACTION_BITS = 55


class PlayerRLSplitter(AbstractPlayer):
    """Plays using the RL-trained MaskablePPO policy."""

    _model = MaskablePPO.load(MODEL_PATH)

    def play(self) -> Meld:
        obs         = self._get_observation()
        action_mask = self._get_action_mask()

        action, _ = self._model.predict(
            obs,
            action_masks=action_mask,
            deterministic=True,
        )

        meld = self._action_to_meld(int(action))

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
