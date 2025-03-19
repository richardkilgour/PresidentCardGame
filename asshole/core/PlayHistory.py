#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

    def __str__(self):
        return ' '.join(c for m in self._memory if m[1] for c in m[1])
