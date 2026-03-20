#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlayerHolder extends PlayerSplitter with a card conservation heuristic.

Strategy:
  - Defer to PlayerSplitter for basic play selection.
  - Additionally, avoid playing a high card if:
      1. The candidate is the highest card in hand, AND
      2. We hold the lowest card still in play (meaning we MUST lead
         at some point, so we need a high card to win that round).
  - Exception: always play instant winners (highest remaining card).

Reasoning: a 3 can only be played as a lead. To lead, you must win a
round. To win a round, you need a high card. So preserve one high card
for each low card you hold.
"""
import logging

from president.core.PlayingCard import PlayingCard
from president.players.PlayerSplitter import PlayerSplitter


class PlayerHolder(PlayerSplitter):

    def play(self):
        candidate = super().play()

        # No target — defer to PlayerSplitter
        if not self.target_meld:
            return candidate

        # Forced pass — no choice
        if not candidate.cards:
            logging.info(f'{self.name}: forced to pass against {self.target_meld}')
            return candidate

        # Instant winner — always play it
        if self.memory.get_highest_remaining() == candidate.cards[0].get_value():
            logging.info(f'{self.name}: {candidate} is a winner, playing it')
            return candidate

        self._log_hand_context(candidate)

        # Conservation heuristic:
        # If the candidate is our highest card, and we hold the lowest
        # card still in play, save the high card for when we need to lead
        if self._should_conserve(candidate):
            logging.info(
                f'{self.name}: conserving {candidate} to protect '
                f'{self._hand[0]} — passing instead'
            )
            return self.possible_plays()[-1]  # Pass

        logging.info(f'{self.name}: playing {candidate} against {self.target_meld}')
        return candidate

    def _should_conserve(self, candidate) -> bool:
        """
        Returns True if we should hold back the candidate and pass instead.

        Conditions (both must be true):
          1. Candidate is the highest card in our hand
          2. We hold the lowest card still in play (need a high card to lead it)
        """
        # Is the candidate our highest card?
        if self._hand[-1].get_value() != candidate.cards[0].get_value():
            return False

        # Do we hold the lowest card still in play?
        lowest_card_value = self._hand[0].get_value()
        for value in range(0, lowest_card_value):
            if self.memory.get_number_remaining(value) > 0:
                # There is a lower card elsewhere — we are not forced to lead
                return False

        # Are there higher cards still out there that we'd rather wait for?
        for value in range(candidate.cards[0].get_value() + 1, 14):
            if self.memory.get_number_remaining(value) > 0:
                return True  # Higher cards exist — don't waste ours yet

        return False

    def _log_hand_context(self, candidate) -> None:
        """Log the cards remaining below and above our hand for debugging."""
        lowest_card = self._hand[0]
        highest_card = self._hand[-1]
        for value in range(0, lowest_card.get_value()):
            remaining = self.memory.get_number_remaining(value)
            logging.debug(
                f'{self.name}: {remaining}x {PlayingCard(value * 4)} remaining '
                f'(below our lowest {lowest_card})'
            )
        for value in range(highest_card.get_value() + 1, 14):
            remaining = self.memory.get_number_remaining(value)
            logging.debug(
                f'{self.name}: {remaining}x {PlayingCard(value * 4)} remaining '
                f'(above our highest {highest_card})'
            )
