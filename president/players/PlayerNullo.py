#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlayerHolder extends PlayerSplitter with a card conservation heuristic.

Strategy:
  - Always play cards lower than max of (the global average, average of hand)
  - Keep the highest cards until they are safe to play (can't be beat)
  - Play out at once.
  - Ignore double and triples for now - just look at the values
"""
import logging
from math import ceil

from president.core.PlayValidator import PlayValidator
from president.core.PlayingCard import PlayingCard
from president.players.PlayerSplitter import PlayerSplitter


class PlayerNullo(PlayerSplitter):

    def play(self, valid_plays):
        candidate = super().play(valid_plays)

        # Forced pass — no choice
        if not candidate.cards:
            logging.info(f'{self.name}: forced to pass')
            return candidate

        # If cards are below this value, just play them
        quartile = 0.75

        # Could consider the global median, but this makes the performance slightly worse
        # threshold =  max(self._get_global_median(quartile), self._get_hand_median(quartile))
        threshold =  self._get_hand_median(quartile)
        if candidate.value() <= threshold:
            return candidate

        # Get rid of singles if there are doubles. Get rid of doubles if there are triples
        sensible_leads = self._get_starting_plays()

        # Take the highest half of the list (round up)
        middle_card = sensible_leads[ceil((len(sensible_leads)-1) * quartile)]

        # See if there are any cards larger than this one
        for value in range(13, middle_card.value(), -1):
            # If not, play the candidate
            remaining = self.memory.get_number_remaining(value)
            my_cards = sum(1 for obj in self._hand if obj.get_value() == value)
            if remaining > my_cards:
                break

        # Must be a pass, right?
        return valid_plays[-1]

    def _get_starting_plays(self):
        # Assume they player will play doubles instead of singles
        all_leads = PlayValidator.possible_plays(self._hand, None)
        result = []
        for i in range(0, len(all_leads) - 1):
            current = all_leads[i]
            # Check if the next meld has the same value
            if all_leads[i + 1].value() != current.value():
                result.append(current)
        # The last one is always added
        result.append(all_leads[-1])
        return result

    def _get_global_median(self, quantile = 0.5):
        # Check the history for each card played
        previous_plays = self.memory.get_previous_melds()
        # Scan the deck to see what's left
        # TODO: Copy of DeckManager.reset_deck()
        temp_deck = [PlayingCard(i) for i in range(54)]
        for meld in previous_plays:
            for card in meld.cards:
                temp_deck.remove(card)
        return temp_deck[ceil((len(temp_deck)-1) * quantile)].get_value()

    def _get_hand_median(self, quantile = 0.5):
        return self._hand[ceil((len(self._hand)-1) * quantile)].get_value()
