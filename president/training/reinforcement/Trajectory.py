#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trajectory captures a complete episode from one player's perspective,
including game state, actions, and metadata.

Rewards are deliberately excluded — the training code decides the
reward function based on metadata.final_rank.

Used for:
  - Supervised learning: extract (state, action) pairs
  - RL training:         derive (state, action, reward, next_state, done)
                         using any reward function applied to final_rank
  - Analysis:            filter by opponent, starting rank, outcome etc.

File format: one JSON record per trajectory, appended to a per-policy
.jsonl file. Human-readable and appendable without loading the full file.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

import numpy as np


RANK_NAMES = ["President", "Vice-President", "Citizen", "Scumbag"]


@dataclass
class OpponentInfo:
    """Information about one opponent in an episode."""
    name: str
    policy: str        # class name e.g. "PlayerHolder", or "anonymous"
    starting_rank: int | None   # 0-3, None if first episode
    final_rank: int             # 0-3

    @property
    def starting_rank_name(self) -> str:
        return RANK_NAMES[self.starting_rank] \
            if self.starting_rank is not None else "—"

    @property
    def final_rank_name(self) -> str:
        return RANK_NAMES[self.final_rank]

    def to_dict(self) -> dict:
        return {
            "name":          self.name,
            "policy":        self.policy,
            "starting_rank": self.starting_rank,
            "final_rank":    self.final_rank,
        }

    @staticmethod
    def from_player(player, memory) -> "OpponentInfo":
        """
        Build from a live player object and the recording player's memory.
        Starting and final ranks are read directly from PlayHistory.
        """
        return OpponentInfo(
            name=player.name,
            policy=player.__class__.__name__,
            starting_rank=memory.starting_position(player),
            final_rank=memory.final_position(player),
        )

    @staticmethod
    def anonymous() -> "OpponentInfo":
        return OpponentInfo(
            name="Unknown",
            policy="anonymous",
            starting_rank=None,
            final_rank=0,
        )

    @staticmethod
    def from_dict(d: dict) -> "OpponentInfo":
        return OpponentInfo(
            name=d["name"],
            policy=d["policy"],
            starting_rank=d.get("starting_rank"),
            final_rank=d["final_rank"],
        )


@dataclass
class TrajectoryMetadata:
    """
    Episode-level metadata for one player's trajectory.

    starting_rank: this player's rank at the start of the episode
                   (from previous episode). None if first episode.
    final_rank:    this player's finishing position (0=President..3=Scumbag)
    opponents:     info about the other three players including their
                   starting and finishing positions.
    """
    policy: str
    player_name: str
    episode_id: str
    timestamp: str
    starting_rank: int | None
    final_rank: int
    opponents: list[OpponentInfo]

    @property
    def starting_rank_name(self) -> str:
        return RANK_NAMES[self.starting_rank] \
            if self.starting_rank is not None else "first episode"

    @property
    def final_rank_name(self) -> str:
        return RANK_NAMES[self.final_rank]

    @staticmethod
    def build(player, opponents: list, memory) -> "TrajectoryMetadata":
        """
        Build metadata at episode end from PlayHistory.

        Args:
            player:    The player whose trajectory this is.
            opponents: Other players in clockwise order.
            memory:    The recording player's PlayHistory — contains
                       starting and final positions for all players.
        """
        return TrajectoryMetadata(
            policy=player.__class__.__name__,
            player_name=player.name,
            episode_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            starting_rank=memory.starting_position(player),
            final_rank=memory.final_position(player),
            opponents=[
                OpponentInfo.from_player(o, memory)
                for o in opponents
            ],
        )

    def to_dict(self) -> dict:
        return {
            "policy":        self.policy,
            "player_name":   self.player_name,
            "episode_id":    self.episode_id,
            "timestamp":     self.timestamp,
            "starting_rank": self.starting_rank,
            "final_rank":    self.final_rank,
            "opponents":     [o.to_dict() for o in self.opponents],
        }

    @staticmethod
    def from_dict(d: dict) -> "TrajectoryMetadata":
        return TrajectoryMetadata(
            policy=d["policy"],
            player_name=d["player_name"],
            episode_id=d["episode_id"],
            timestamp=d["timestamp"],
            starting_rank=d.get("starting_rank"),
            final_rank=d["final_rank"],
            opponents=[
                OpponentInfo.from_dict(o) for o in d["opponents"]
            ],
        )


@dataclass
class Trajectory:
    """
    Complete episode trajectory from one player's perspective.

    states:   (T, 341) binary array — game state at each decision point
    actions:  (T, 54)  binary array — meld played at each decision point
    done:     (T,)     bool array   — True on player's final turn

    Rewards are not stored — apply any reward function at training time
    using metadata.final_rank.
    """
    metadata: TrajectoryMetadata
    states:   np.ndarray    # shape (T, 341)
    actions:  np.ndarray    # shape (T, 54)
    done:     np.ndarray    # shape (T,)

    # -------------------------------------------------------------------------
    # Supervised learning
    # -------------------------------------------------------------------------

    def to_supervised(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract (states, actions) pairs for supervised learning.

        Returns:
            states:  (T, 341)
            actions: (T, 54)
        """
        return self.states, self.actions

    # -------------------------------------------------------------------------
    # RL
    # -------------------------------------------------------------------------

    def to_rl(self, rank_rewards: dict[int, float] | None = None
              ) -> tuple[np.ndarray, np.ndarray,
                         np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract (states, actions, rewards, next_states, dones) for RL.

        Args:
            rank_rewards: Maps final_rank to reward value.
                          Defaults to {0: 2.0, 1: 1.0, 2: -1.0, 3: -2.0}

        Returns:
            states:      (T-1, 341)
            actions:     (T-1, 54)
            rewards:     (T-1,)  — episode reward on final transition only
            next_states: (T-1, 341)
            dones:       (T-1,)
        """
        if rank_rewards is None:
            rank_rewards = {0: 2.0, 1: 1.0, 2: -1.0, 3: -2.0}

        T = len(self.states)
        rewards = np.zeros(T, dtype=np.float32)
        rewards[-1] = rank_rewards.get(self.metadata.final_rank, 0.0)

        return (
            self.states[:-1],
            self.actions[:-1],
            rewards[:-1],
            self.states[1:],
            self.done[:-1],
        )

    # -------------------------------------------------------------------------
    # Serialisation
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "states":   self.states.tolist(),
            "actions":  self.actions.tolist(),
            "done":     self.done.tolist(),
        }

    @staticmethod
    def from_dict(d: dict) -> "Trajectory":
        return Trajectory(
            metadata=TrajectoryMetadata.from_dict(d["metadata"]),
            states=np.array(d["states"],  dtype=np.int8),
            actions=np.array(d["actions"], dtype=np.int8),
            done=np.array(d["done"],       dtype=bool),
        )

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def __str__(self) -> str:
        opponents = ", ".join(
            f"{o.name} ({o.policy})"
            for o in self.metadata.opponents
        )
        return (
            f"Trajectory [{self.metadata.episode_id[:8]}] "
            f"{self.metadata.player_name} ({self.metadata.policy}) | "
            f"{self.metadata.starting_rank_name} → "
            f"{self.metadata.final_rank_name} | "
            f"T={len(self.states)} turns | "
            f"opponents: {opponents}"
        )
