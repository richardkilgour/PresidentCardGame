#!/usr/bin/env python
# -*- coding: utf-8 -*-
from asshole.core.Meld import Meld


class PlayHistory:
    def __init__(self):
        self._memory = []

    def clear(self):
        self._memory = []

    def add_play(self, player, meld):
        # TODO: This is None if someone played out high, and everyone passed
        # meld.cards throws
        if meld:
            self._memory.append((player, meld))

    def get_highest_remaining(self):
        for x in range(13, 0, -1):
            if self.get_number_remaining(x):
                return x
        return -1

    def get_number_remaining(self, value):
        """Calculate the number of cards remaining for the given value"""
        # Start with total possible (2 jokers or 4 regular cards)
        starting_count = 2 if value == 13 else 4

        # Subtract cards that have been played
        played = sum(len(m[1]) for m in self._memory
                     if m[1] and m[1][0].get_value() == value)

        return starting_count - played

    def previous_plays_generator(self, start_players, ignore_latest=True):
        """
        Generate previous play indices in reverse chronological order.

        Args:
            start_players: List of players, with the previous player at index 0
            ignore_latest: Whether to ignore the most recent play in memory

        Yields:
            Integer indices representing previous plays (54 for pass, others for actual plays)
        """
        # Make a copy of players that we can modify
        current_players = start_players.copy()

        # Determine where to start in the memory
        memory = self._memory
        if ignore_latest:
            memory = memory[:-1]  # Skip the last entry

        # Iterate through memory in reverse order
        for move in reversed(memory):
            # Rotate players until we find the one who made this move
            while move[0] != current_players[0]:
                # Not the expected player - Assume they passed
                yield Meld()  # Pass code
                # Bring the last player to the front
                current_players = [current_players[-1]] + current_players[:-1]

            # We found the player who made the move
            other_meld = move[1]
            yield other_meld
            # Bring the last player to the front
            current_players = [current_players[-1]] + current_players[:-1]


    def __str__(self):
        return ' '.join(c for m in self._memory if m[1] for c in m[1])
