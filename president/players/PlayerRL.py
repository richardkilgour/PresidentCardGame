#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlayerRL — a generic RL-trained player using MaskablePPO.

Takes an experiment name and loads the best_model.zip saved by EvalCallback.
Uses direct policy network inference for speed — bypasses SB3 overhead.
Falls back to the first legal play if the model predicts an illegal action.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from president.core.AbstractPlayer import AbstractPlayer
from president.core.Meld import Meld
from president.models.single_meld_mlp import encode_state

EXPERIMENTS_DIR = (Path(__file__).resolve().parent.parent / "training" / "experiments").resolve()

ACTION_BITS = 55

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlayerRL(AbstractPlayer):
    """Plays using an RL-trained MaskablePPO policy."""

    def __init__(self, name: str, model: str):
        super().__init__(name)
        exp_dir    = (EXPERIMENTS_DIR / model).resolve()
        model_path = exp_dir / "best_model.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"RL model not found: {model_path}")
        self._device = _device
        self._model  = MaskablePPO.load(model_path, device=_device)

    def play(self, valid_plays) -> Meld:
        obs  = encode_state(self._hand, self.target_meld).astype(np.float32)
        mask = self._get_action_mask(valid_plays)

        with torch.no_grad():
            obs_tensor  = torch.tensor(obs,  dtype=torch.float32).unsqueeze(0).to(self._device)
            mask_tensor = torch.tensor(mask, dtype=torch.bool   ).unsqueeze(0).to(self._device)

            distribution = self._model.policy.get_distribution(obs_tensor)
            logits       = distribution.distribution.logits.clone()
            logits[~mask_tensor] = float("-inf")
            action = logits.argmax(dim=1).item()

        meld = self._action_to_meld(action)
        if meld in valid_plays:
            return meld

        logging.warning(
            f"{self.name}: RL model predicted illegal meld, falling back. "
            f"Hand: {self._hand}, target: {self.target_meld}"
        )
        return valid_plays[0]

    def _get_action_mask(self, valid_plays) -> np.ndarray:
        mask = np.zeros(ACTION_BITS, dtype=bool)
        for meld in valid_plays:
            mask[self._meld_to_action(meld)] = True
        return mask

    def _action_to_meld(self, action: int) -> Meld:
        if action == 54:
            return Meld()
        value      = action // 4
        count      = action % 4 + 1
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
