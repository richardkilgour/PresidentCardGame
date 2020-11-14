#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The second most simple player type
Most simple would play self.possible_plays()[0]
This one will always play the lowest possible card unless it would split a set.
"""
import logging
from asshole.player.AbstractPlayer import AbstractPlayer


class PlayerSimple(AbstractPlayer):
    def play(self):
        """
        Given a list of cards, choose a set to play
        If no minimum, just play the lowest card or lowest set of cards
        Return a list of cards, or None if the desire is to pass
        """

        # We know the target meld, and play the lowest option
        selection = self.possible_plays()

        # If pass is the only remaining option, take it
        if len(selection) == 1:
            return selection[0]

        for s in selection[:-1]:
            # If this is not a set, just play it
            if self.number_of_cards_of_value(s.cards[0].value) == len(s.cards):
                return s
            logging.info("Found a set of {} x {}, so not playing {}".format(len(s.cards), s.cards[0], s.cards))

        # Will get here is all the possible plays are doubles - pass
        return selection[-1]
