#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Loads a RL model and uses the default history to construct the inputs to the model
Checks if the output is valid before returning the best card(s) to play.
"""
import torch

from president.RL.data_utils import hand_to_indices, meld_to_index, index_to_meld
from president.RL.grid_model import load_model
from president.core.AbstractPlayer import AbstractPlayer

def masked_argmax(output_probs, valid_indices):
    """
    Selects the index with the highest probability, but only among valid indices.

    Args:
        output_probs (torch.Tensor): Model output tensor of shape (batch_size, 55).
        valid_indices (list or torch.Tensor): List of valid indices.

    Returns:
        torch.Tensor: The selected class indices for each batch element.
    """
    # Convert valid_indices to a mask (batch_size, 55) where invalid indices are -inf
    mask = torch.full_like(output_probs, float('-inf'))
    mask[:, valid_indices] = output_probs[:, valid_indices]

    # Take argmax only on valid indices
    return mask.argmax(dim=1)


class RLSimple(AbstractPlayer):
    def __init__(self, name, model_file = None):
        super().__init__(name)
        # Default model location. Move to config?
        if not model_file:
            model_file = "RL/best_model.pt"
        self.trained_model = load_model(filepath = model_file)

    def play(self):
        """
        Given a list of cards, choose a set to play
        If no minimum, just play the lowest card or lowest set of cards
        Return a list of cards, or None if the desire is to pass
        """
        super().play()
        # We know the target meld, and play the lowest option that beats the meld
        possible_plays = self.possible_plays()

        # Create an input vector for the model (similar to DataGrabber)
        # Get up to 3 previous plays
        prev_plays = []
        for _, play, desc in self.memory.previous_plays_generator():
            cards = meld_to_index(play)
            prev_plays.insert(0, cards)
            if len(prev_plays) >= 3:
                break
        hand = hand_to_indices(self._hand)
        padding = [54] * (14-len(hand))
        net_in = torch.tensor([prev_plays + hand + padding],  dtype=torch.int64)
        net_out = self.trained_model(net_in)

        # Turn possible plays into a mask
        valid_indices = [meld_to_index(p) for p in possible_plays]
        # Filter the output by the possible plays
        best_play = net_out.argmax()
        best_meld = masked_argmax(net_out, valid_indices)
        if best_play != best_meld.item():
            print(f"Target: {self.target_meld} Hand: {' '.join([c.__str__() for c in self._hand])}")
            print(f"Wanted to play {index_to_meld(best_play)}, but decided to play {index_to_meld(best_meld.item())}")
        meld = index_to_meld(best_meld)
        return meld
