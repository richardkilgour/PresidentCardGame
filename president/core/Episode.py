#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Episode module for controlling a single episode of a card game.

Takes players in their current positions and plays through to new positions.

Responsible for: state machine flow, turn management, rank assignment,
and notifying listeners of all game events.

Not responsible for: card movement mechanics (CardHandler),
serialisation (GameCheckpoint - future).
"""
from __future__ import annotations

import logging
from enum import Enum

from president.core.AbstractPlayer import AbstractPlayer
from president.core.CardGameListener import CardGameListener
from president.core.CardHandler import CardHandler
from president.core.DeckManager import DeckManager
from president.core.Meld import Meld
from president.core.PlayValidator import PlayValidator
from president.core.PlayerManager import PlayerManager


class State(Enum):
    INITIALISED = 1
    DEALING = 2
    SWAPPING = 3
    ROUND_STARTING = 4
    PLAYING = 5
    HAND_WON = 6
    FINISHED = 7


class Episode:
    def __init__(self, player_manager: PlayerManager, ranks: list[AbstractPlayer],
                 deck: DeckManager, listener_list: list[CardGameListener]) -> None:
        self.state: State = State.INITIALISED
        self.player_manager = player_manager
        self.ranks = ranks
        # '␆' represents "not played yet"; external dependency, refactor separately
        self.current_melds = ['␆', '␆', '␆', '␆']
        self.active_players = []
        self.deck = deck
        self.listener_list = listener_list
        self.card_handler = CardHandler(deck)

    def clear(self):
        self.state: State = State.INITIALISED
        self.player_manager = None
        self.current_melds = ['␆', '␆', '␆', '␆']
        self.active_players = []
        self.deck = []
        self.listener_list = []

    def notify_listeners(self, notify_func_name: str, *args) -> None:
        for p in self.listener_list:
            getattr(p, notify_func_name)(*args)

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def deal(self) -> None:
        """Deal cards to all players. Asserts clean state before dealing."""
        for player in self.player_manager.players:
            assert player.report_remaining_cards() == 0, \
                f"{player.name} has {player.report_remaining_cards()} remaining cards before dealing."
        assert len(self.card_handler.discard_pile) == 0, \
            "Discards pile is not empty before dealing."
        self.deck.deal_cards(self.player_manager.players)

    def swap_cards(self) -> None:
        """Swap cards between players based on their previous episode rankings."""
        self.card_handler.swap_for_new_episode(self.ranks)
        if len(self.ranks) == 4:
            self.notify_listeners("notify_cards_swapped", self.ranks[0], self.ranks[3], 2)
            self.notify_listeners("notify_cards_swapped", self.ranks[1], self.ranks[2], 1)

    def move_to_front(self, front_player: AbstractPlayer) -> None:
        """Rotate the players list until the specified player is at the front."""
        while self.player_manager.players[0] != front_player:
            self.player_manager.players.append(self.player_manager.players.pop(0))

    def pick_round_starter(self) -> None:
        """Set the starting player and notify listeners that a new hand is beginning."""
        if len(self.ranks) == 4:
            starting_player = self.ranks[3]  # Scumbag starts
        else:
            starting_player = self.player_manager.find_card_holder(0, 0)  # Holder of 3♠
        self.move_to_front(starting_player)
        self.notify_listeners("notify_hand_start")
        self.notify_listeners("notify_player_turn", self.player_manager.players[0])

    # -------------------------------------------------------------------------
    # Turn management
    # -------------------------------------------------------------------------

    def target_meld(self) -> Meld | None:
        """Return the highest meld currently on the table, or None if none played yet."""
        highest_meld = None
        for m in self.current_melds:
            if m and m != '␆' and (highest_meld is None or m > highest_meld):
                highest_meld = m
        return highest_meld

    def get_players_with_cards(self) -> list[AbstractPlayer]:
        """Return all players who still have cards in hand."""
        active_players = []
        for player in self.player_manager.players:
            if player.report_remaining_cards() == 0:
                assert player in self.ranks, f"{player.name} should have a rank."
            else:
                active_players.append(player)
        return active_players

    def set_player_finished(self, player: AbstractPlayer) -> None:
        """Assign a rank to a player who has played out and notify listeners."""
        ranking = len(self.ranks)
        self.ranks.append(player)
        logging.info(f"{player.name} played out and is ranked {player.ranking_names[ranking]}")
        player.set_position(ranking)
        self.notify_listeners("notify_played_out", player, ranking)
        player_names = ' '.join(p.name if p else '' for p in self.ranks)
        logging.info(f"Positions: {player_names}")

    def player_turn(self) -> None:
        """Execute a single turn for the current active player."""
        assert self.active_players, "No active players available for turn."
        logging.info(f"Players who have not passed = {' '.join(x.name for x in self.active_players)}")

        if len(self.active_players) == 1:
            self.set_player_finished(self.active_players[0])
            return

        current_target = self.target_meld()
        logging.info(f'Currently played highest card = {current_target}')

        player = self.active_players[0]
        if player.report_remaining_cards() == 0:
            self.active_players.remove(player)
            return

        self.notify_listeners("notify_player_turn", player)

        action = player.play()
        if action == '␆':  # no-op, async player not ready yet
            return

        PlayValidator.validate(player, action, current_target)

        if not action.cards:
            self.notify_listeners("notify_pass", player)
            self.active_players.remove(player)
        else:
            self.notify_listeners("notify_play", player, action)
            self.card_handler.play_meld(player, action)
            index = self.player_manager.players.index(player)
            self.current_melds[index] = action
            if player.report_remaining_cards() == 0:
                self.set_player_finished(player)
            self.active_players.append(self.active_players.pop(0))

    # -------------------------------------------------------------------------
    # Post-episode cleanup
    # -------------------------------------------------------------------------

    def post_episode_checks(self) -> None:
        """Finalise the episode: log rankings, collect scumbag cards, restore deck."""
        assert len(self.ranks) == 4
        self.card_handler.collect_scumbag_cards(self.ranks[-1])
        self.card_handler.restore_deck()
        assert not any(pos is None for pos in self.ranks), \
            "There should be no empty positions."
        for i, p in enumerate(self.ranks):
            logging.info(f"{p.name} is ranked as {p.ranking_names[i]}")
        for p in self.player_manager.players:
            assert p.report_remaining_cards() == 0, \
                f"{p.name} still has cards remaining."

    # -------------------------------------------------------------------------
    # State machine
    # -------------------------------------------------------------------------

    def step(self) -> list[AbstractPlayer | None]:
        """
        Advance the episode state machine by one step.

        Returns:
            Current rankings. Empty until episode is finished.
        """
        if self.state == State.INITIALISED:
            self.deck.shuffle()
            self.state = State.DEALING
            self.deal()
        if self.state == State.DEALING:
            self.state = State.SWAPPING
            for player in self.player_manager.players:
                logging.debug(f'{player.name} has {player}')
            self.swap_cards()
        if self.state == State.SWAPPING:
            self.state = State.ROUND_STARTING
            self.pick_round_starter()
            self.ranks = []
        if self.state == State.ROUND_STARTING:
            self.active_players = self.get_players_with_cards()
            self.current_melds = ['␆', '␆', '␆', '␆']
            # Notify players sitting out this hand
            for player in self.player_manager.players:
                if player not in self.active_players:
                    self.notify_listeners("notify_waiting", player)
            self.state = State.PLAYING
        if self.state == State.PLAYING:
            self.player_turn()
            if len(self.active_players) == 1:
                self.state = State.HAND_WON
        if self.state == State.HAND_WON:
            if len(self.ranks) == 4:
                self.post_episode_checks()
                self.state = State.FINISHED
            else:
                assert len(self.active_players) == 1, \
                    "Expected one active player in HAND_WON state."
                self.notify_listeners("notify_hand_won", self.active_players[0])
                self.move_to_front(self.active_players[0])
                self.state = State.ROUND_STARTING

        return self.ranks
