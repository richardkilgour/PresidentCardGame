#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EpisodeRecord is a CardGameListener that records a full episode trajectory
for use in RL training.

It captures (state, action, reward) tuples for each player turn, then
assigns rewards retrospectively when final rankings are known.

Supports SARSA, Deep-Q, and recurrent models. Attach to GameMaster like
any other listener — it is invisible to game logic.

Usage:
    recorder = EpisodeRecord(players)
    game_master.add_listener(recorder)
    # ... run episode ...
    dataset = recorder.to_dataset()
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from president.core.CardGameListener import CardGameListener
from president.core.Meld import Meld


# Rewards mirror AbstractPlayer.get_score() per-episode contribution
RANK_REWARDS = {
    0: 2.0,   # President
    1: 1.0,   # Vice-President
    2: -1.0,  # Citizen
    3: -2.0,  # Scumbag
}


@dataclass
class Transition:
    player: Any                     # Player reference
    state: np.ndarray               # Encoded game state before action
    action: int                     # Encoded action (card index or pass)
    next_state: np.ndarray | None   # Encoded state after action; None until filled
    reward: float                   # 0.0 until episode ends, then filled retrospectively
    done: bool                      # True on the player's final transition


class EpisodeRecord(CardGameListener):
    """
    Records a full episode trajectory for RL training.
    Attaches as a listener — no changes to game logic required.
    """

    # Index used to represent a pass action
    PASS_ACTION = 54

    def __init__(self, players: list) -> None:
        """
        Args:
            players: The list of players in seat order. Used to identify
                     which transitions belong to which player.
        """
        super().__init__()
        self.players = players
        self._transitions: list[Transition] = []
        self._pending: dict[Any, Transition] = {}  # player -> last incomplete transition

    def clear(self) -> None:
        """Reset for a new episode. Called automatically on notify_hand_start."""
        self._transitions = []
        self._pending = {}

    # -------------------------------------------------------------------------
    # Listener interface
    # -------------------------------------------------------------------------

    def notify_hand_start(self) -> None:
        # Do NOT call super() — we never clear our memory mid-episode
        # unlike CardGameListener which clears per hand
        pass

    def notify_play(self, player, meld: Meld) -> None:
        self._record_action(player, meld)

    def notify_pass(self, player) -> None:
        self._record_action(player, Meld())

    def notify_played_out(self, player, rank: int) -> None:
        """Assign reward to all of this player's transitions when their rank is known."""
        reward = RANK_REWARDS.get(rank, 0.0)
        for transition in self._transitions:
            if transition.player is player:
                transition.reward = reward
                if rank == len(self.players) - 1:
                    transition.done = True
        # Close any pending transition for this player
        if player in self._pending:
            self._pending[player].done = True
            self._transitions.append(self._pending.pop(player))

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _encode_state(self, player) -> np.ndarray:
        """
        Encode the current observable state for a player.
        Uses the player's own encode() which captures hand and meld status.
        """
        return player.encode()

    def _encode_action(self, meld: Meld) -> int:
        """
        Encode a meld as a single integer action.
        Uses the index of the first card, or PASS_ACTION for a pass.
        """
        if not meld.cards:
            return self.PASS_ACTION
        return meld.cards[0].get_index()

    def _record_action(self, player, meld: Meld) -> None:
        """
        Record a (state, action) pair. Closes the previous pending transition
        for this player by filling in next_state, then opens a new one.
        """
        current_state = self._encode_state(player)
        action = self._encode_action(meld)

        # Close the previous transition for this player
        if player in self._pending:
            self._pending[player].next_state = current_state
            self._transitions.append(self._pending.pop(player))

        # Open a new pending transition
        self._pending[player] = Transition(
            player=player,
            state=current_state,
            action=action,
            next_state=None,
            reward=0.0,
            done=False,
        )

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    def to_dataset(self) -> list[Transition]:
        """
        Return the completed trajectory for this episode.
        Excludes any transitions still pending (should be none after episode ends).
        """
        incomplete = [t for t in self._transitions if t.next_state is None]
        if incomplete:
            import warnings
            warnings.warn(
                f"EpisodeRecord.to_dataset() called with {len(incomplete)} "
                "incomplete transitions. Was the episode finished?",
                RuntimeWarning
            )
        return [t for t in self._transitions if t.next_state is not None]

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the trajectory as stacked numpy arrays, ready for training.

        Returns:
            states, actions, rewards, next_states, dones
        """
        dataset = self.to_dataset()
        if not dataset:
            raise RuntimeError("No completed transitions to return.")
        states      = np.stack([t.state for t in dataset])
        actions     = np.array([t.action for t in dataset], dtype=np.int32)
        rewards     = np.array([t.reward for t in dataset], dtype=np.float32)
        next_states = np.stack([t.next_state for t in dataset])
        dones       = np.array([t.done for t in dataset], dtype=bool)
        return states, actions, rewards, next_states, dones