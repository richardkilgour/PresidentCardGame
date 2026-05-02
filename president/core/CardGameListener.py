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

    def _view_of(self, player):
        """Return the representation of player to store in this listener's memory.

        Subclasses may override to substitute a restricted view (e.g. PlayerView)
        for opponents, ensuring they are never exposed as full AbstractPlayers.
        """
        return player

    @property
    def players(self):
        return self.memory.players

    @players.setter
    def players(self, value):
        self.memory.set_players(
            [self._view_of(p) if p is not None else None for p in value]
        )

    def notify_player_joined(self, new_player, position):
        self.memory.add_player(position, self._view_of(new_player))

    def notify_game_stated(self):
        pass

    def notify_hand_start(self):
        self.memory.clear()

    def notify_hand_won(self, winner):
        self.memory.add_round_won(self._view_of(winner))

    def notify_played_out(self, player, rank):
        self.memory.add_player_finished(self._view_of(player), rank)

    def notify_play(self, player, meld):
        self.memory.add_play(self._view_of(player), meld)

    def notify_pass(self, player):
        self.memory.add_play(self._view_of(player), Meld())

    def notify_waiting(self, player):
        self.memory.add_waiting(self._view_of(player))

    def notify_player_turn(self, player):
        pass

    def notify_cards_swapped(self, player_good, player_bad, num_cards, cards_to_good=None, cards_to_bad=None):
        self.memory.record_swap(
            self._view_of(player_good),
            self._view_of(player_bad),
            num_cards,
        )

    def notify_player_replaced(self, old_player, new_player):
        self.memory.replace_player(
            self._view_of(old_player),
            self._view_of(new_player),
        )

    def notify_illegal_play(self, player, action, reason: str):
        pass

    def notify_episode_end(self, final_ranks: list, starting_ranks: list) -> None:
        pass
