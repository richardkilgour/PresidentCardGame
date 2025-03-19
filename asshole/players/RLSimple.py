#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The second most simple players type
Most simple would play possible_plays(self.target_meld)[0]
This one will always play the lowest possible card _unless_ it would split a set.
"""
import logging

from asshole.RL.data_utils import hand_to_indices, meld_to_index
from asshole.RL.file_utils import load_model
from asshole.core.AbstractPlayer import AbstractPlayer


class RLSimple(AbstractPlayer):
    def __init__(self, model_file, name):
        super().__init__(name)
        self.trained_model = load_model(filepath = model_file)

    """Concrete players with a simple and stupid strategy - play the lowest possible card"""
    def play(self):
        """
        Given a list of cards, choose a set to play
        If no minimum, just play the lowest card or lowest set of cards
        Return a list of cards, or None if the desire is to pass
        """
        super().play()
        # We know the target meld, and play the lowest option that beats the meld
        possible_plays = self.possible_plays(self.target_meld)

        # TODO: Mostly duplicetd from DataGrabber
        previous_player_index = (self.players.index(self) + 3) % 4
        # Move previous player to the front
        players = self.players[previous_player_index:] + self.players[:previous_player_index]

        # Create an input vector for the model (similar to DataGrabber)
        # Get up to 3 previous plays
        prev_plays = []
        for play in self.memory.previous_plays_generator(players):
            cards = meld_to_index(play)
            prev_plays.append(cards)
            if len(prev_plays) >= 3:
                break
        hand = hand_to_indices(self._hand)
        padding = [54] * (14-len(hand))
        self.input = prev_plays + hand + padding
