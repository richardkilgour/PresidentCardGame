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
        self.players = [None, None, None, None]
        self.player_status = ['Absent', 'Absent', 'Absent', 'Absent']

    def notify_player_joined(self, new_player, position):
        self.players[position] = new_player
        self.player_status[position] = 'Waiting'
        self.memory.set_players(self.players)

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

    def opponents_clockwise(self, player=None) -> list:
        """
        Return the other players in clockwise order relative to the given player.
        If player is None, uses self as the reference point.
        Can be called by any listener — not just the player themselves.

        Args:
            player: The player to use as the reference point.
                    Defaults to self if None.

        Returns:
            List of the other players in clockwise order.

        Raises:
            ValueError: If the player is not seated at this table.
        """
        reference = player if player is not None else self
        if reference not in self.players:
            raise ValueError(
                f"{getattr(reference, 'name', reference)} is not seated "
                f"at this table. "
                f"Seated players: "
                f"{[p.name for p in self.players if p is not None]}"
            )
        seat = self.players.index(reference)
        n = len(self.players)
        return [
            self.players[(seat + i) % n]
            for i in range(1, n)
            if self.players[(seat + i) % n] is not None
        ]
