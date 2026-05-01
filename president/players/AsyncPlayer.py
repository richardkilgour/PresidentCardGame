#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dummy player that just waits for a call to play, and plays whatever was buffered
"""
import logging

from president.core.AbstractPlayer import AbstractPlayer
from president.core.Meld import Meld


class AsyncPlayer(AbstractPlayer):
    def __init__(self, name: str):
        AbstractPlayer.__init__(self, name)
        self.card_to_play = '␆'

    def add_play(self, play: Meld):
        if self.card_to_play != '␆':
            logging.warning(f'{self.name}: buffered play {self.card_to_play} dropped, replaced by {play}')
        self.card_to_play = play

    def play(self, valid_plays):
        card_to_play = self.card_to_play
        self.card_to_play = '␆'
        return card_to_play

    def notify_hand_won(self, winner):
        super().notify_hand_won(winner)
        self.card_to_play = '␆'  # discard any stale buffered play between hands
