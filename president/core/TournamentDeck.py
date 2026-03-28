#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TournamentDeck wraps DeckManager to provide controlled dealing
for tournament play. Instead of shuffling and dealing randomly,
it deals preset hands and supports hand rotation across seats.
"""
from __future__ import annotations

import random
from president.core.DeckManager import DeckManager
from president.core.PlayingCard import PlayingCard


class TournamentDeck(DeckManager):
    """
    A DeckManager that deals preset hands instead of shuffling.

    Usage:
        deck = TournamentDeck()
        deck.new_deal()          # fresh random hands, rotation=0
        deck.rotate()            # advance rotation before next episode
    """

    def __init__(self, deck_size: int = 54, seed: int = None):
        self._preset_hands: list[list[PlayingCard]] | None = None
        super().__init__(deck_size)
        self._rotation: int = 0
        if seed:
            random.seed(seed)

    def new_deal(self) -> None:
        """
        Generate a fresh random set of 4 hands and store as preset.
        Resets rotation to 0. Call once per position combination.
        """
        cards = [PlayingCard(i) for i in range(54)]
        random.shuffle(cards)
        hands = [[], [], [], []]
        for i, card in enumerate(cards):
            hands[i % 4].append(card)
        for hand in hands:
            hand.sort(key=lambda c: c.get_index())
        self._preset_hands = hands
        self._rotation = 0

    def rotate(self) -> None:
        """
        Advance hand rotation by one seat.
        Call before each of the 3 remaining rotated episodes.
        """
        self._rotation = (self._rotation + 1) % 4

    def shuffle(self) -> None:
        """Suppress shuffle — preset hands are set via new_deal()."""
        if self._preset_hands is None:
            # Allow shuffle during __init__ before preset is established
            super().shuffle()

    def deal_cards(self, players: list) -> None:
        """
        Deal preset hands to players respecting current rotation.
        Player i receives hand[(i + rotation) % 4].

        Clears self.deck first so CardHandler.restore_deck() sees
        an empty deck, consistent with normal deal behaviour.
        """
        if self._preset_hands is None:
            super().deal_cards(players)
            return

        # Normal deal_cards() empties self.deck via pop() —
        # do the same so CardHandler.restore_deck() assertion passes
        self.deck.clear()

        for i, player in enumerate(players):
            hand = self._preset_hands[(i + self._rotation) % 4]
            for card in hand:
                player.card_to_hand(card)
