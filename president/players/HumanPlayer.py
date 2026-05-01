#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HumanPlayer is the base class for all locally-controlled human players.

Extends AbstractPlayer with a single additional capability: the ability to
quit the game. Quitting raises QuitGame (a BaseException), which propagates
naturally through all `except Exception` handlers in Episode and GameMaster
without being swallowed, and is caught by whoever is driving the game loop.

AI players cannot quit. Human players can.
"""
from __future__ import annotations

from president.core.AbstractPlayer import AbstractPlayer


class QuitGame(BaseException):
    """Raised by a human player to signal they want to quit."""
    def __init__(self, player: "HumanPlayer | None" = None) -> None:
        self.player = player
        super().__init__(f"{player.name} quit the game." if player else "Player quit.")


class HumanPlayer(AbstractPlayer):
    """
    Abstract base for all human-controlled players.

    Subclasses must implement play(). When the human wants to quit,
    they call self.request_quit(), which raises QuitGame.
    The game loop (offline.py, game_wrapper, etc.) is responsible for
    catching QuitGame and handling any saves.
    """

    def request_quit(self) -> None:
        """Signal that this player wants to quit. Never returns."""
        raise QuitGame(self)
