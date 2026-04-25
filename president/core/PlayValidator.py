#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlayValidator validates a meld returned by player.play() before it is
accepted by Episode.

Catches:
  - Wrong return type (not a Meld)
  - If you lead a round, you may not pass
  - Cards not in the player's hand
  - Cards of mixed values in the meld
  - Meld does not beat the current target
"""
from collections import Counter

from president.core.IllegalPlayError import IllegalPlayError
from president.core.Meld import Meld


class PlayValidator:

    @staticmethod
    def validate(player, action, current_target, must_include_index=None) -> None:
        """
        Validate a play. Raises IllegalPlayError if invalid.

        Args:
            player:              The player who made the play.
            action:              The value returned by player.play().
            current_target:      The current highest meld, or None if no meld yet.
            must_include_index:  Card index that must appear in the meld, or None.
        """
        PlayValidator._check_type(player, action)
        PlayValidator._check_must_play(player, action, current_target)
        PlayValidator._check_cards_in_hand(player, action)
        PlayValidator._check_consistent_value(player, action)
        PlayValidator._check_beats_target(player, action, current_target)
        PlayValidator._check_must_include(player, action, must_include_index)

    @staticmethod
    def _check_must_play(player, action, current_target):
        if not action.cards:
            # Passing is only valid when there is a target meld to beat.
            # When current_target is None the player leads and must play a card.
            if current_target is None:
                raise IllegalPlayError(
                    player, action,
                    "Must lead with a card — passing is not allowed when you have won the round"
                )
            return

    @staticmethod
    def _check_type(player, action) -> None:
        if not isinstance(action, Meld):
            raise IllegalPlayError(
                player, action,
                f"play() must return a Meld, got {type(action).__name__}"
            )

    @staticmethod
    def _check_cards_in_hand(player, action) -> None:
        available = Counter(player.get_hand_indices())
        for card in action.cards:
            available[card.get_index()] -= 1
            if available[card.get_index()] < 0:
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
    def _check_must_include(player, action, must_include_index) -> None:
        if must_include_index is None or not action.cards:
            return
        if must_include_index not in [c.get_index() for c in action.cards]:
            raise IllegalPlayError(
                player, action,
                f"Opening play must include the card at index {must_include_index}"
            )

    @staticmethod
    def _check_beats_target(player, action, current_target) -> None:
        if not action.cards:
            return  # pass is validated by _check_must_play
        if current_target and action <= current_target:
            raise IllegalPlayError(
                player, action,
                f"{action} does not beat current target {current_target}"
            )

    # ─────────────────────────────────────────────
    # Hand-level helpers (no player context needed)
    # ─────────────────────────────────────────────

    @staticmethod
    def possible_plays(hand: list, target, must_include_index=None) -> list:
        """Return all legal melds for the given hand and target.

        Args:
            hand:                The player's current hand (list of PlayingCard).
            target:              The current highest meld, or None if leading.
            must_include_index:  Card index that must appear in the meld, or None.
        """
        candidates = PlayValidator._generate_candidates(hand)
        plays = [m for m in candidates if PlayValidator._is_legal(m, target)]
        if must_include_index is not None:
            plays = [m for m in plays
                     if m.cards and must_include_index in [c.get_index() for c in m.cards]]
        return plays

    @staticmethod
    def _generate_candidates(hand: list) -> list:
        """All singles, pairs, triples, quads from hand, plus pass."""
        candidates = []
        for card in hand:
            if candidates and card == candidates[-1].cards[0]:
                candidates.append(Meld(card, candidates[-1]))
            else:
                candidates.append(Meld(card))
        candidates.append(Meld())
        return candidates

    @staticmethod
    def _is_legal(meld, target) -> bool:
        """True if meld is legal given the current target (None = leading)."""
        if not meld.cards:
            return target is not None  # pass only when there is something to beat
        return not (target and meld <= target)
