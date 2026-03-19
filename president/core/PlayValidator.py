#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlayValidator validates a meld returned by player.play() before it is
accepted by Episode.

Catches:
  - Wrong return type (not a Meld)
  - Cards not in the player's hand
  - Cards of mixed values in the meld
  - Meld does not beat the current target
"""
from president.core.IllegalPlayError import IllegalPlayError
from president.core.Meld import Meld


class PlayValidator:

    @staticmethod
    def validate(player, action, current_target) -> None:
        """
        Validate a play. Raises IllegalPlayError if invalid.

        Args:
            player:         The player who made the play.
            action:         The value returned by player.play().
            current_target: The current highest meld, or None if no meld yet.
        """
        PlayValidator._check_type(player, action)

        # Pass is always valid — no further checks needed
        if not action.cards:
            return

        PlayValidator._check_cards_in_hand(player, action)
        PlayValidator._check_consistent_value(player, action)
        PlayValidator._check_beats_target(player, action, current_target)

    @staticmethod
    def _check_type(player, action) -> None:
        if not isinstance(action, Meld):
            raise IllegalPlayError(
                player, action,
                f"play() must return a Meld, got {type(action).__name__}"
            )

    @staticmethod
    def _check_cards_in_hand(player, action) -> None:
        hand_indices = player.get_hand_indices()
        for card in action.cards:
            if card.get_index() not in hand_indices:
                raise IllegalPlayError(
                    player, action,
                    f"Card {card} is not in {player.name}'s hand"
                )

    @staticmethod
    def _check_consistent_value(player, action) -> None:
        # Jokers (value 13) may appear alongside other cards in special combos
        # so only check consistency among non-joker cards
        non_joker_values = {
            c.get_value() for c in action.cards
            if c.get_value() < 13
        }
        if len(non_joker_values) > 1:
            raise IllegalPlayError(
                player, action,
                f"Meld contains cards of mixed values: {non_joker_values}"
            )

    @staticmethod
    def _check_beats_target(player, action, current_target) -> None:
        if current_target and action <= current_target:
            raise IllegalPlayError(
                player, action,
                f"{action} does not beat current target {current_target}"
            )
