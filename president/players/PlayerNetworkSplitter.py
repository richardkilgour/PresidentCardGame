#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import logging
import numpy as np
import torch
import torch.nn as nn

from president.core.AbstractPlayer import AbstractPlayer
from president.core.Meld import Meld
from president.core.StateEncoder import StateEncoder

MODEL_PATH = Path(__file__).parent.parent / "models" / "player_splitter_mlp.pt"
MELD_BITS  = 54

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model() -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(108, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 55),  # 0-53 melds, 54 = pass
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=_device,
                                     weights_only=True))
    model.eval()
    return model.to(_device)


class PlayerNetworkSplitter(AbstractPlayer):
    """Plays using a trained MLP clone of PlayerSplitter."""

    _device = _device
    _model  = _build_model()

    def play(self) -> Meld:
        hand_enc   = StateEncoder._encode_hand(self._hand)
        target_enc = StateEncoder.encode_meld(self.target_meld) \
                     if self.target_meld is not None \
                     else np.zeros(MELD_BITS, dtype=np.int8)

        x = torch.tensor(
            np.concatenate([hand_enc, target_enc]),
            dtype=torch.float32,
            device=self._device,
        ).unsqueeze(0)

        with torch.no_grad():
            pred = self._model(x).argmax(dim=1).item()

        # Class 54 = pass
        if pred == 54:
            return Meld()

        # Decode meld bit position to value + count
        if pred >= 52:
            value = 13  # joker
            count = pred - 52 + 1
        else:
            value = pred // 4
            count = pred % 4 + 1

        predicted_meld = self._meld_from_hand(value, count)

        legal = self.possible_plays()
        if predicted_meld is not None and predicted_meld in legal:
            return predicted_meld

        logging.warning(
            f"{self.name}: network predicted illegal meld "
            f"(value={value}, count={count}), "
            f"falling back. Hand: {self._hand}, target: {self.target_meld}"
        )
        return legal[0]

    def _meld_from_hand(self, value: int, count: int) -> Meld | None:
        """
        Build a Meld of `count` cards of `value` using actual cards from _hand.
        Returns None if the hand doesn't contain enough cards of that value.
        """
        candidates = [c for c in self._hand if c.get_value() == value]
        if len(candidates) < count:
            return None
        meld = None
        for card in candidates[:count]:
            meld = Meld(card, meld) if meld else Meld(card)
        return meld
