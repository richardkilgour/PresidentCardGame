#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A Card Game Listener is aware of all the goings-on in the game, and by default keeps a history of them.
"""
from president.core.Meld import Meld
from president.core.PlayHistory import PlayHistory


class CardGameListener:
    def __init__(self):
        self.memory = PlayHistory()

    @property
    def players(self):
        return self.memory.players

    @players.setter
    def players(self, value):
        self.memory.set_players(value)

    def notify_player_joined(self, new_player, position):
        self.memory.add_player(position, new_player)

    def notify_game_stated(self):
        pass

    def notify_hand_start(self):
        self.memory.clear()

    def notify_hand_won(self, winner):
        self.memory.add_round_won(winner)

    def notify_played_out(self, player, rank):
        # Someone just finished by getting rid of all their cards
        self.memory.add_player_finished(player, rank)

    def notify_play(self, player, meld):
        # Someone played cards
        self.memory.add_play(player, meld)

    def notify_pass(self, player):
        # Someone passed
        self.memory.add_play(player, Meld())

    def notify_waiting(self, player):
        self.memory.add_waiting(player)

    def notify_player_turn(self, player):
        pass

    def notify_cards_swapped(self, player_good, player_bad, num_cards, cards_to_good=None, cards_to_bad=None):
        # Notify that one player swapped cards with another.
        # player_good got good card(s) from player_bad and vice versa.
        # PlayHistory infers starting positions (President/VP/Citizen/Scumbag) from these calls.
        self.memory.record_swap(player_good, player_bad, num_cards)

    def notify_illegal_play(self, player, action, reason: str):
        # A player attempted an illegal play
        pass

    def notify_episode_end(self, final_ranks: list, starting_ranks: list) -> None:
        pass

