#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A Card Game Listener is aware of all the goings-on in the game, and by default keeps a history of them
"""
from president.core.PlayHistory import PlayHistory


class CardGameListener:
    def __init__(self):
        self.memory = PlayHistory()
        self.players = [None, None, None, None]
        self.player_status = ['Absent', 'Absent', 'Absent', 'Absent']

    def notify_player_joined(self, new_player, position):
        # In case anyone wants to track their opponents
        self.players[position] = new_player
        self.player_status[position] = 'Waiting'

    def notify_game_stated(self):
        pass

    def notify_hand_start(self):
        self.memory.clear()

    def notify_hand_won(self, winner):
        pass

    def notify_played_out(self, opponent, pos):
        # Someone just lost all their cards
        pass

    def notify_play(self, player, meld):
        # Someone is just about to play cards, or pass
        self.memory.add_play(player, meld)

    def notify_player_turn(self, player):
        # It is this player's turn to play
        pass

    def notify_cards_swapped(self, player_good, player_bad, num_cards):
        # Notify that one player swapped cards with another.
        # player_good got good card(s) from player_bad and vice versa
        pass