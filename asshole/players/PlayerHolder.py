#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Will save a high card for each low card
Otherwise acts as a splitter
"""
import logging

from asshole.core.PlayingCard import PlayingCard
from asshole.players.PlayerSplitter import PlayerSplitter


class PlayerHolder(PlayerSplitter):
    def play(self):
        """
        Given a list of cards, choose a set to play
        If no minimum, just play the lowest card(s)
        Return a list of cards, or None if the desire is to pass
        """
        # The card we would play if boing by the parent strategy
        candidate = super().play()

        if not self.target_meld:
            return candidate

        # Find the possible melded, and play the lowest one, unless we have a double

        # If pass is the only option, then do that
        if len(self.possible_plays(self.target_meld)) == 1 and not candidate.cards:
            logging.info("{}'s reaction to the {} is a mandatory {}".format(self.name, self.target_meld, candidate))
            return candidate

        # Instant winner? Play for sure!
        if candidate.cards and self.memory.get_highest_remaining() == candidate.cards[0].get_value:
            logging.info("{} knows {} is a winner".format(self.name, candidate))
            return candidate

        # Debug only block
        if False:
            lowest_card = self._hand[0]
            logging.debug("{}'s lowest card is {}".format(self.name, lowest_card))
            for x in range(0, lowest_card.get_value()):
                logging.debug("{} knows there are {} x {} out there".format(self.name, self.memory.get_number_remaining(x), PlayingCard(x * 4)))
            highest_card = self._hand[-1]
            logging.debug("{}'s highest card is {}".format(self.name, highest_card))
            for x in range(highest_card.get_value() + 1, 14):
                logging.debug("{} knows there are {} x {} out there".format(self.name, self.memory.get_number_remaining(x), PlayingCard(x * 4)))

        # Do we want to wait for a higher card to be played?
        # 1) Do we have the lowest possible card? No - then play, otherwise
        # 2) We should only play our highest card if it's a winner
        if (self._hand[-1].get_value() == candidate.cards[0].get_value()):
            save_it = True
            lowest_card = self._hand[0]
            for x in range(0, lowest_card.get_value()):
                if self.memory.get_number_remaining(x) > 0:
                    # There is at least one lower card, so we can play
                    save_it = False
                    break
            if save_it:
                for x in range(candidate.cards[0].get_value() + 1, 14):
                    if self.memory.get_number_remaining(x) > 0:
                        # Don't risk it!
                        logging.info("{} decides not to play {} to protect the {}".format(self.name, candidate, lowest_card))
                        candidate = self.possible_plays[-1]
                        break

        logging.info("{}'s reaction to the {} is a {}".format(self.name, self.target_meld, candidate))
        return candidate