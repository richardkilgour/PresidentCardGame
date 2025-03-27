#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The second most simple players type
Most simple would play possible_plays()[0]
This one will always play the lowest possible card _unless_ it would split a set.
"""
import torch

from president.RL.data_utils import hand_to_indices, meld_to_index, index_to_meld
from president.RL.file_utils import load_model
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

    """Concrete players with a simple and stupid strategy - play the lowest possible card"""
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
        best_meld = masked_argmax(net_out, valid_indices)
        meld = index_to_meld(best_meld)
        return meld
