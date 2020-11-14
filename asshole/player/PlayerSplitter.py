#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Will not split a double or triple except 2 and Joker)
Also will split the highest cards to stay in the game
"""
import logging

from asshole.player.PlayerSimple import PlayerSimple


class PlayerSplitter(PlayerSimple):
    def play(self):
        """
        Given a list of cards, choose a set to play
        If no minimum, just play the lowest card(s)
        Return a list of cards, or None if the desire is to pass
        """
        my_meld = super().play()

        if not self.target_meld:
            logging.info("No minimum value, so {} is going to play {}".format(self.name, my_meld))
            return my_meld

        # Find the possible melds, and play the lowest one, unless we have a double
        # GM knows the current target
        selection = self.possible_plays()

        for candidate in selection:
            if not candidate.cards:
                # If pass is the remaining option, take it
                break
            if candidate.cards[0].value() >= 12:
                # Just play 2s and Jokers
                break
            # Otherwise do not break longer runs
            if not self.will_split(candidate):
                break

        # Fallback option - we can play but are considering to pass
        # Split the highest card to stay in the hand
        if not candidate.cards and (len(selection) > 1):
            logging.info("{} decides to split a {}".format(self.name, selection[-2]))
            candidate = selection[-2]

        logging.info("{}'s reaction to the {} is a {}".format(self.name, self.target_meld, candidate))
        return candidate
