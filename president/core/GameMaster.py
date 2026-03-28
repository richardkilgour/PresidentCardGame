#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GameMaster controls the overall flow of the card game across multiple rounds.

Responsibilities: player registration, round lifecycle, listener management,
illegal play handling, and end-of-tournament statistics.

Not responsible for: turn-by-turn game logic (Episode),
card movement mechanics (CardHandler).
"""
from __future__ import annotations

import logging
from enum import Enum

from president.core.AbstractPlayer import AbstractPlayer
from president.core.CardGameListener import CardGameListener
from president.core.DeckManager import DeckManager
from president.core.Episode import Episode, State
from president.core.GameCheckpoint import GameCheckpoint
from president.core.IllegalPlayError import IllegalPlayError
from president.core.PlayerManager import PlayerManager
from president.core.PlayerRegistry import PlayerRegistry

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class IllegalPlayPolicy(Enum):
    TERMINATE  = "terminate"   # Save checkpoint, raise, stop the game
    DISQUALIFY = "disqualify"  # Save checkpoint, replace with fallback, continue
    PENALISE   = "penalise"    # Force a pass, continue (no checkpoint, for RL)


class GameMaster:

    def __init__(self, deck_size: int = 54,
                 registry: PlayerRegistry = None,
                 policy: IllegalPlayPolicy = IllegalPlayPolicy.DISQUALIFY,
                 fallback_player_name: str = None) -> None:
        """
        Args:
            deck_size:            Number of cards in the deck.
            registry:             PlayerRegistry used to create players.
                                  A fresh empty registry is created if not provided.
            policy:               How to handle illegal plays.
            fallback_player_name: Registered name of the player type to substitute
                                  on disqualification. Defaults to the first
                                  registered entry if not specified.
        """
        self.player_manager = PlayerManager()
        self.positions = []
        self.deck = DeckManager(deck_size)
        self.listener_list = []
        self.number_of_rounds = None
        self.round_number = 0
        self.episode = None
        self.registry = registry or PlayerRegistry()
        self.policy = policy
        self.fallback_player_name = fallback_player_name
        self._checkpoint: GameCheckpoint | None = None

    def set_checkpoint(self, checkpoint: GameCheckpoint) -> None:
        self._checkpoint = checkpoint

    # -------------------------------------------------------------------------
    # Listener management
    # -------------------------------------------------------------------------

    def add_listener(self, listener: CardGameListener) -> None:
        self.listener_list.append(listener)

    def notify_listeners(self, notify_func_name: str, *args) -> None:
        for listener in self.listener_list:
            getattr(listener, notify_func_name)(*args)

    # -------------------------------------------------------------------------
    # Player management
    # -------------------------------------------------------------------------

    def notify_player_joined(self, player: AbstractPlayer, position: int) -> None:
        """Notify all existing listeners of the new player, then add as a listener."""
        self.notify_listeners("notify_player_joined", player, position)
        for i, existing_player in enumerate(self.player_manager.players):
            if existing_player:
                player.notify_player_joined(existing_player, i)
        self.add_listener(player)

    def add_player(self, player: AbstractPlayer,
                   position: int | None = None) -> int:
        """
        Add an already-constructed player to the game.

        Args:
            player:   The player to add.
            position: Optional seat index.

        Returns:
            The seat index assigned.
        """
        position = self.player_manager.add_player(player, position)
        self.notify_player_joined(player, position)
        return position

    def make_player(self, type_name: str,
                    player_name: str = None) -> AbstractPlayer:
        """
        Create a player from the registry and add them to the game.

        Args:
            type_name:   Registered type name e.g. "Naive", "RL_v2".
            player_name: Name for this player instance. Defaults to type_name.

        Returns:
            The newly created and seated player.
        """
        player = self.registry.create(type_name, player_name)
        self.add_player(player)
        return player

    # -------------------------------------------------------------------------
    # Game lifecycle
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """
        Create a new Episode, passing rankings from the previous one
        so card swapping and turn order are set correctly.
        """
        logger.info(f"--- Start of a new Episode --- {self.positions=}")
        self.deck.shuffle()
        self.episode = Episode(
            self.player_manager, self.positions, self.deck, self.listener_list
        )

    def start(self, number_of_rounds: int = 100, positions=None) -> None:
        """
        Start a series of episodes.

        Args:
            number_of_rounds: Number of episodes to play. None for infinite.
            positions: Starting positions (Default - no positions)

        Raises:
            Exception: If any seat is unfilled.
        """
        if positions is None:
            positions = []
        if None in self.player_manager.players:
            raise Exception(
                f"Not enough players — all {len(self.player_manager.players)} seats must be filled."
            )
        self.round_number = 0
        self.number_of_rounds = number_of_rounds
        self.positions = positions
        self.reset()
        self.notify_listeners("notify_game_stated")

    def on_round_completed(self) -> bool:
        """
        Finalise the current episode and set up the next one.

        Returns:
            True if the game is over, False otherwise.
        """
        self.notify_listeners(
            "notify_episode_end",
            self.episode.ranks,
            self.positions,
        )
        self.positions = self.episode.ranks
        logger.info(f"--- Episode Finished with positions {self.positions} ---")
        self.round_number += 1
        if self.number_of_rounds and self.round_number >= self.number_of_rounds:
            #print(self.position_stats_str())
            #self.remove_worst_player()
            return True
        self.reset()
        return False

    def step(self) -> bool:
        """
        Advance the game by one step.

        Returns:
            True if the tournament is finished, False otherwise.
        """
        if self.episode is None:
            raise RuntimeError("Game has not been started. Call start() first.")
        try:
            if self.episode.state == State.FINISHED:
                return self.on_round_completed()
            self.episode.step()
            return False
        except IllegalPlayError as e:
            return self._handle_illegal_play(e)
        except Exception as e:
            logger.error(f"Unexpected error during step: {e}", exc_info=True)
            if self._checkpoint:
                self._checkpoint.save_on_error(
                    GameCheckpoint.stamped_path("crash")
                )
            raise

    # -------------------------------------------------------------------------
    # Illegal play handling
    # -------------------------------------------------------------------------

    def _handle_illegal_play(self, error: IllegalPlayError) -> bool:
        """
        Handle an illegal play according to the current policy.
        Always logs. Saves a checkpoint for TERMINATE and DISQUALIFY.

        Returns:
            True if the game should stop, False if it can continue.
        """
        logger.error(f"Illegal play: {error}")
        self.notify_listeners(
            "notify_illegal_play", error.player, error.action, error.reason
        )

        if self.policy == IllegalPlayPolicy.TERMINATE:
            if self._checkpoint:
                self._checkpoint.save_on_error(
                    GameCheckpoint.stamped_path("illegal_play")
                )
            raise error

        elif self.policy == IllegalPlayPolicy.DISQUALIFY:
            if self._checkpoint:
                self._checkpoint.save_on_error(
                    GameCheckpoint.stamped_path("illegal_play")
                )
            self._disqualify_player(error.player)
            return False

        elif self.policy == IllegalPlayPolicy.PENALISE:
            logger.warning(f"{error.player.name} penalised — forcing a pass.")
            self._force_pass(error.player)
            return False

    def _disqualify_player(self, player: AbstractPlayer) -> None:
        """
        Replace a disqualified player with the fallback type from the registry.
        Transfers the hand to the replacement.
        """
        fallback_name = self.fallback_player_name or self.registry.names()[0]
        fallback = self.registry.create(fallback_name, f"{player.name}[auto]")

        seat = self.player_manager.players.index(player)
        fallback._hand = player._hand[:]
        self.player_manager.players[seat] = fallback

        if player in self.episode.active_players:
            i = self.episode.active_players.index(player)
            self.episode.active_players[i] = fallback

        if player in self.listener_list:
            i = self.listener_list.index(player)
            self.listener_list[i] = fallback

        logger.warning(
            f"{player.name} disqualified at seat {seat}, "
            f"replaced with {fallback_name}."
        )
        self.notify_listeners("notify_player_joined", fallback, seat)

    def _force_pass(self, player: AbstractPlayer) -> None:
        """Force a pass for a player who made an illegal play."""
        if player in self.episode.active_players:
            self.episode.active_players.remove(player)
        self.notify_listeners("notify_pass", player)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def position_stats_str(self) -> str:
        """Return a formatted summary of each player's position counts and score."""
        result = []
        for player in self.player_manager.players:
            if player is None:
                continue
            rank_parts = '; '.join(
                f'{player.ranking_names[i]} {count}'
                for i, count in enumerate(player.position_count)
            )
            result.append(f'{player.name}: {rank_parts}. Score = {player.get_score()}')
        return '\n'.join(result)

    def remove_worst_player(self) -> None:
        """Remove the player with the lowest cumulative score."""
        worst_player = min(
            (p for p in self.player_manager.players if p is not None),
            key=lambda p: p.get_score()
        )
        logger.info(f'{worst_player.name} is pissed off and quits')
        self.player_manager.players.remove(worst_player)
