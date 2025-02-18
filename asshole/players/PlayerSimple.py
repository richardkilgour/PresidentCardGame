#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The second most simple players type
Most simple would play possible_plays(self._hand, self.target_meld, self.name)[0]
This one will always play the lowest possible card unless it would split a set.
"""
import logging
from asshole.core.AbstractPlayer import AbstractPlayer


class PlayerSimple(AbstractPlayer):
    """Concrete players with a simple and stupid strategy - play the lowest possible card"""
    def play(self):
        """
        Given a list of cards, choose a set to play
        If no minimum, just play the lowest card or lowest set of cards
        Return a list of cards, or None if the desire is to pass
        """
        super().play()
        # We know the target meld, and play the lowest option

        # If pass is the only remaining option, take it
        if len(self.possible_plays) == 1:
            return self.possible_plays[0]

        for s in self.possible_plays[:-1]:
            # If this is not a set, just play it (use will_split()?)
            if self.number_of_cards_of_value(s.cards[0].get_value()) == len(s.cards):
                return s
            logging.info(f'Found a set of {len(s.cards)} x {s.cards[0]}, so not playing {s.cards}')

        # Will get here is all the possible plays are doubles - pass
        return self.possible_plays[-1]
