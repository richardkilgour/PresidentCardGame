#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlayerNetwork — a generic network player.

Loads the model class and weights from an experiment directory.
The experiment name (e.g. "splitter_clone_v1") is the only required argument
beyond the player name; everything else is derived from the experiment's
config.json and the conventionally-named .pt file.

Experiment directory layout (under training/experiments/<model>/):
    config.json          — must contain a "model": {"class": "..."} entry
    <model>.pt           — trained weights
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import torch

from president.core.AbstractPlayer import AbstractPlayer
from president.core.Meld import Meld

EXPERIMENTS_DIR = (Path(__file__).resolve().parent.parent / "training" / "experiments").resolve()

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlayerNetwork(AbstractPlayer):
    """Plays using a trained network loaded from an experiment directory."""

    def __init__(self, name: str, model: str):
        super().__init__(name)
        exp_dir = (EXPERIMENTS_DIR / model).resolve()
        if not exp_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

        config_path = exp_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {exp_dir}")
        with open(config_path) as f:
            cfg = json.load(f)

        model_cfg = cfg.get("model")
        if model_cfg is None or "class" not in model_cfg:
            raise ValueError(f"config.json in {exp_dir} must contain a 'model.class' entry")

        dotted = model_cfg["class"]
        module_path, class_name = dotted.rsplit(".", 1)
        module    = importlib.import_module(module_path)
        model_cls = getattr(module, class_name)

        weights_path = exp_dir / f"{exp_dir.name}.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        architecture = model_cfg.get("architecture", {})
        self._model  = model_cls(**architecture)
        self._model.load_state_dict(
            torch.load(weights_path, map_location=_device, weights_only=True)
        )
        self._model.eval()
        self._model.to(_device)
        self._device    = _device
        self._model_cls = model_cls

    def play(self) -> Meld:
        x = torch.tensor(
            self._model_cls.encode_state(self.memory, self),
            dtype=torch.float32,
            device=self._device,
        ).unsqueeze(0)

        mask = torch.tensor(
            self._action_mask(), dtype=torch.bool, device=self._device
        ).unsqueeze(0)

        with torch.no_grad():
            logits = self._model(x)
            logits[~mask] = float("-inf")
            pred = logits.argmax(dim=1).item()

        return self._action_to_meld(pred)

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(55, dtype=bool)
        for meld in self.possible_plays():
            mask[self._meld_to_action(meld)] = True
        return mask

    def _action_to_meld(self, action: int) -> Meld:
        if action == 54:
            return Meld()
        value = action // 4
        count = action % 4 + 1
        candidates = [c for c in self._hand if c.get_value() == value]
        meld = None
        for card in candidates[:count]:
            meld = Meld(card, meld) if meld else Meld(card)
        return meld

    @staticmethod
    def _meld_to_action(meld: Meld) -> int:
        if not meld.cards:
            return 54
        return meld.cards[0].get_value() * 4 + len(meld.cards) - 1
