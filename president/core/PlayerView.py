#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Callable


class PlayerView:
    """Public face of a player as seen by opponents.

    Holds only the player's name and a callable that returns their
    current card count — no reference to the underlying player object.
    """

    def __init__(
        self,
        name: str,
        player_type: str,
        remaining_cards_fn: Callable[[], int],
        previous_position_fn: Callable[[], "int | None"],
        current_position_fn: Callable[[], "int | None"],
    ):
        self.name = name
        self.player_type = player_type
        self._remaining_cards_fn = remaining_cards_fn
        self._previous_position_fn = previous_position_fn
        self._current_position_fn = current_position_fn

    def report_remaining_cards(self) -> int:
        return self._remaining_cards_fn()

    def previous_position(self) -> "int | None":
        """Rank index from the previous episode (0=President … 3=Scumbag), or None if first game."""
        return self._previous_position_fn()

    def current_position(self) -> "int | None":
        """Rank index achieved so far this episode, or None if still playing."""
        return self._current_position_fn()

    def __repr__(self):
        return f"PlayerView({self.name!r}, {self.player_type})"
