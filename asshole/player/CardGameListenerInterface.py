#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A Card Game Listener is aware of all the goings on in the game, and keeps a history of them
"""
from asshole.PlayHistory import PlayHistory


class CardGameListenerInterface:
    def __init__(self):
        self.memory = PlayHistory()
        self.opponents = []

    def notify_player_joined(self, new_player):
        # In case anyone wants to track their opponent
        if new_player is not self:
            self.opponents.append(new_player)

    def notify_hand_start(self, starter):
        self.memory.clear()

    def notify_hand_won(self, winner):
        pass

    def notify_played_out(self, opponent, pos):
        # Someone just lost all their cards
        pass

    def notify_play(self, player, meld):
        # Someone just played (or passed)
        self.memory.add_play(player, meld)