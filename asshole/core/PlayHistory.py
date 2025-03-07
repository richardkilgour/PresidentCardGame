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
        if value == 13:
            num_remaining = 2
        else:
            num_remaining = 4
        for m in self._memory:
            if m[1]:
                if m[1][0].get_value() == value:
                    num_remaining -= len(m[1])
        return num_remaining

    def __str__(self):
        # Should probably dump raw, but instead sort it
        str = ""
        for m in self._memory:
            if m[1]:
                for c in m[1]:
                    str += "{} ".format(c)
        return str