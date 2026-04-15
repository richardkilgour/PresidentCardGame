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
from collections import deque
from enum import Enum

from president.core.AbstractPlayer import AbstractPlayer
from president.core.CardGameListener import CardGameListener
from president.core.CardHandler import CardHandler
from president.core.DeckManager import DeckManager
from president.core.HandIntegrityChecker import HandIntegrityChecker
from president.core.Meld import Meld
from president.core.PlayValidator import PlayValidator
from president.core.PlayerManager import PlayerManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class State(Enum):
    INITIALISED = 1
    DEALING = 2
    SWAPPING = 3
    ROUND_STARTING = 4
    PLAYING = 5
    HAND_WON = 6
    FINISHED = 7


class Episode:
    def __init__(self, player_manager: PlayerManager, starting_ranks: list[AbstractPlayer],
                 deck: DeckManager, listener_list: list[CardGameListener],
                 card_handler: CardHandler) -> None:
        self.state: State = State.INITIALISED
        self.player_manager = player_manager
        # starting_ranks: rankings from the previous episode, used for card swapping
        # and to determine who starts. Empty list on the very first episode.
        self.starting_ranks: list[AbstractPlayer] = starting_ranks
        # ranks: accumulated finishing order for the current episode, built up as
        # players play out. Empty until the episode completes.
        self.ranks: list[AbstractPlayer] = []
        # '␆' represents "not played yet"; external dependency, refactor separately
        self.current_melds: list = ['␆'] * self.player_manager.player_count
        self.active_players: list[AbstractPlayer] = []
        self.deck = deck
        self.listener_list = listener_list
        self.card_handler = card_handler

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
        n = len(self.starting_ranks)
        self.card_handler.swap_for_new_episode(self.starting_ranks)
        # Notify listeners for each symmetric swap pair: rank 0 swaps with rank n-1,
        # rank 1 swaps with rank n-2, etc. Only notify for the upper half to avoid
        # duplicate notifications.
        for i in range(n // 2):
            j = n - 1 - i
            num_cards = (n // 2) - i
            self.notify_listeners("notify_cards_swapped",
                                  self.starting_ranks[i], self.starting_ranks[j], num_cards)

    def move_to_front(self, front_player: AbstractPlayer) -> None:
        """Rotate the players deque until the specified player is at the front."""
        players = deque(self.player_manager.players)
        # index() is O(n) but rotate() is O(k) — avoids repeated pop(0)/append
        idx = players.index(front_player)
        players.rotate(-idx)
        self.player_manager.players = list(players)

    def pick_round_starter(self) -> None:
        """Set the starting player and notify listeners that a new hand is beginning."""
        n = len(self.starting_ranks)
        if n > 0:
            starting_player = self.starting_ranks[-1]  # Lowest rank (scumbag) starts
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
        """Return all players who still have cards in hand.

        Invariant: any player with no cards at this point must already have a rank,
        meaning they finished in a previous hand of this episode.
        """
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
        if player in self.ranks:
            raise AssertionError(
                f"Rank integrity failed for {player.name} "
                f" already in {[str(c) for c in self.ranks]}"
            )
        self.ranks.append(player)
        logger.info(f"{player.name} played out and is ranked {player.ranking_names[ranking]}")
        player.set_position(ranking)
        self.notify_listeners("notify_played_out", player, ranking)
        player_names = ' '.join(p.name if p else '' for p in self.ranks)
        logger.info(f"Positions: {player_names}")

    def player_turn(self) -> None:
        """Execute a single turn for the current active player.

        Note: when only one active player remains, they are declared finished
        immediately without calling notify_player_turn or play(). This is intentional
        — the last player has no meaningful choice to make — but means listeners will
        not receive a turn notification for that player's final "turn".
        """
        assert self.active_players, "No active players available for turn."
        logger.info(f"Players who have not passed = {' '.join(x.name for x in self.active_players)}")

        if len(self.active_players) == 1:
            self.set_player_finished(self.active_players[0])
            return

        current_target = self.target_meld()
        logger.info(f'Currently played highest card = {current_target}')

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
        """Finalise the episode: verify integrity, collect scumbag cards, restore deck."""
        n = self.player_manager.player_count
        assert len(self.ranks) == n, \
            f"Expected {n} ranked players, got {len(self.ranks)}."
        HandIntegrityChecker.verify(self.player_manager, self.ranks)
        self.card_handler.collect_scumbag_cards(self.ranks[-1])
        self.card_handler.restore_deck()
        assert not any(pos is None for pos in self.ranks), \
            "There should be no empty positions."
        for i, p in enumerate(self.ranks):
            logger.info(f"{p.name} is ranked as {p.ranking_names[i]}")
        for p in self.player_manager.players:
            assert p.report_remaining_cards() == 0, \
                f"{p.name} still has cards remaining."

    # -------------------------------------------------------------------------
    # State machine
    # -------------------------------------------------------------------------

    def step(self) -> list[AbstractPlayer]:
        """
        Advance the episode state machine by one step.

        Each call advances exactly one state. PLAYING may be called many times
        (once per player turn) before transitioning to HAND_WON.

        Returns:
            Current rankings (self.ranks). Empty until the episode is finished.
        """
        if self.state == State.INITIALISED:
            self.deck.shuffle()
            self.state = State.DEALING

        elif self.state == State.DEALING:
            self.deal()
            for player in self.player_manager.players:
                logger.debug(f'Start: {player}')
            self.state = State.SWAPPING

        elif self.state == State.SWAPPING:
            self.swap_cards()
            for player in self.player_manager.players:
                logger.debug(f'Sw: {player}')
            self.pick_round_starter()
            self.state = State.ROUND_STARTING

        elif self.state == State.ROUND_STARTING:
            self.active_players = self.get_players_with_cards()
            self.current_melds = ['␆'] * self.player_manager.player_count
            # Notify players sitting out this hand
            for player in self.player_manager.players:
                if player not in self.active_players:
                    self.notify_listeners("notify_waiting", player)
            self.state = State.PLAYING

        elif self.state == State.PLAYING:
            self.player_turn()
            if len(self.active_players) == 1:
                self.state = State.HAND_WON

        elif self.state == State.HAND_WON:
            if len(self.ranks) == self.player_manager.player_count:
                self.post_episode_checks()
                self.state = State.FINISHED
            else:
                assert len(self.active_players) == 1, \
                    "Expected one active player in HAND_WON state."
                self.notify_listeners("notify_hand_won", self.active_players[0])
                self.move_to_front(self.active_players[0])
                self.state = State.ROUND_STARTING

        return self.ranks
