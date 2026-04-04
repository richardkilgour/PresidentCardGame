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
import logging
from pathlib import Path

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

        # Load model class from experiment config
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

        # Load weights
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
            self._model_cls.encode_state(self._hand, self.target_meld),
            dtype=torch.float32,
            device=self._device,
        ).unsqueeze(0)

        with torch.no_grad():
            pred = self._model(x).argmax(dim=1).item()

        if pred == 54:
            return Meld()

        if pred >= 52:
            value = 13  # joker
            count = pred - 52 + 1
        else:
            value = pred // 4
            count = pred % 4 + 1

        predicted_meld = self._meld_from_hand(value, count)
        legal          = self.possible_plays()

        if predicted_meld is not None and predicted_meld in legal:
            return predicted_meld

        logging.warning(
            f"{self.name}: network predicted illegal meld "
            f"(value={value}, count={count}), falling back. "
            f"Hand: {self._hand}, target: {self.target_meld}"
        )
        return legal[0]

    def _meld_from_hand(self, value: int, count: int) -> Meld | None:
        candidates = [c for c in self._hand if c.get_value() == value]
        if len(candidates) < count:
            return None
        meld = None
        for card in candidates[:count]:
            meld = Meld(card, meld) if meld else Meld(card)
        return meld
