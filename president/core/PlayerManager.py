#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlayerManager owns the player registry and seat assignments.

Responsibilities:
  - Track which player sits in which seat
  - Construct new players
  - Find which player holds a specific card

Not responsible for: card movement, game state, or player status.
Those concerns belong to CardHandler and Episode respectively.
"""
from __future__ import annotations

import logging

from president.core.AbstractPlayer import AbstractPlayer

logger = logging.getLogger(__name__)


class PlayerManager:
    def __init__(self, capacity: int = 4) -> None:
        self.players: list[AbstractPlayer | None] = [None] * capacity

    @property
    def player_count(self) -> int:
        """Return the number of seats occupied by an actual player."""
        return sum(1 for p in self.players if p is not None)

    def add_player(self, player: AbstractPlayer, position: int | None = None) -> int:
        """
        Add a player to the first available seat, or a specific seat if given.

        Args:
            player: The player to add.
            position: Optional seat index. Uses first empty seat if not given.

        Returns:
            The seat index the player was assigned to, or -1 on failure.
        """
        if position is None:
            position = self.players.index(None)
        if self.players[position]:
            logger.warning("Can't replace existing player at position=%d", position)
            return -1
        self.players[position] = player
        return position

    def swap_player(self, old_player: AbstractPlayer, new_player: AbstractPlayer) -> int:
        """
        Replace old_player with new_player in their seat.

        Returns:
            The seat index of the replaced player.
        """
        seat = self.players.index(old_player)
        self.players[seat] = new_player
        return seat

    def find_card_holder(self, target_value: int, target_suit: int) -> AbstractPlayer | None:
        """
        Find the player holding a card with the given value and suit.
        Assumes hands are sorted in ascending value order.

        Args:
            target_value: The card value to find.
            target_suit: The card suit to find.

        Returns:
            The player holding the card, or None if not found.
        """
        for player in self.players:
            if player is None:
                continue
            for card in player._hand:
                if card.get_value() > target_value:
                    break
                if card.get_value() == target_value and card.get_suit() == target_suit:
                    return player
        return None
