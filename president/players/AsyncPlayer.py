#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dummy player that just waits for a call to play, and plays whatever was buffered
"""
from president.core.AbstractPlayer import AbstractPlayer
from president.core.Meld import Meld


class AsyncPlayer(AbstractPlayer):
    def __init__(self, name: str):
        AbstractPlayer.__init__(self, name)
        self.card_to_play = '␆'

    def add_play(self, play: Meld):
        self.card_to_play = play

    def play(self):
        card_to_play = self.card_to_play
        self.card_to_play = '␆'
        return card_to_play
