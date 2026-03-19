#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The simplest possible AI player.
Always plays the lowest available meld, even if it splits a set.
"""
from president.core.AbstractPlayer import AbstractPlayer


class PlayerNaive(AbstractPlayer):
    """Plays the lowest possible meld unconditionally."""

    def play(self):
        possible_plays = self.possible_plays()
        return possible_plays[0]