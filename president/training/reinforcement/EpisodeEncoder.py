#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EpisodeEncoder builds a Trajectory from a player's PlayHistory
at the end of an episode.

Walks the PlayHistory, calls StateEncoder at each decision point,
and assembles the full (state, action, done) sequence.

Rewards are not stored — apply any reward function at training time
using metadata.final_rank.

All state is derived from PlayHistory alone — no live player state
is used except name and class for metadata.

Starting and final positions are read directly from PlayHistory
(populated by notify_cards_swapped and add_player_finished) — they
no longer need to be passed in as separate arguments.

Usage:
    encoder = EpisodeEncoder()
    trajectory = encoder.encode(player=player, opponents=opponents)
    TrajectoryStore.append(trajectory)
"""
import numpy as np

from president.core.PlayHistory import EventType, PlayHistory
from president.training.data.StateEncoder import StateEncoder, MELD_BITS, TOTAL_BITS
from president.training.reinforcement.Trajectory import Trajectory, TrajectoryMetadata


class EpisodeEncoder:
    """
    Builds a Trajectory from a player's PlayHistory at episode end.

    Reads entirely from PlayHistory — no live player state is used.
    One EpisodeEncoder can be reused across many episodes and players.
    """

    def __init__(self) -> None:
        self._state_encoder = StateEncoder()

    def encode(self, player, opponents: list) -> Trajectory:
        """
        Encode the full episode from one player's perspective.

        Args:
            player:    The player whose trajectory to encode.
                       Only player.memory (PlayHistory) and
                       player identity are used — no live state.
            opponents: Other players in clockwise order (for metadata).

        Returns:
            A complete Trajectory ready for saving or training.

        Raises:
            RuntimeError: If no decision points found in history.
        """
        history = player.memory
        decision_points = self._find_decision_points(player, history)

        if not decision_points:
            raise RuntimeError(
                f"EpisodeEncoder: no decision points found for "
                f"{player.name}. Was the episode completed?"
            )

        T = len(decision_points)
        states  = np.zeros((T, TOTAL_BITS), dtype=np.int8)
        actions = np.zeros((T, MELD_BITS),  dtype=np.int8)
        done    = np.zeros(T,               dtype=bool)

        for i, (event_index, event) in enumerate(decision_points):
            # Encode state from history up to and including this event.
            # event.hand captures the hand before cards were removed.
            # No live player state is used.
            history_slice = history._memory[:event_index + 1]
            states[i]  = self._state_encoder.encode_from_event(
                event, history_slice
            )
            actions[i] = StateEncoder.encode_meld(event.meld)

        # Mark the player's final turn as done
        done[-1] = True

        metadata = TrajectoryMetadata.build(
            player=player,
            opponents=opponents,
            memory=history,
        )

        return Trajectory(
            metadata=metadata,
            states=states,
            actions=actions,
            done=done,
        )

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    @staticmethod
    def _find_decision_points(player,
                              history: PlayHistory) -> list[tuple[int, any]]:
        """
        Find all events where this player made a decision (played or passed).
        Returns (event_index, event) pairs so history can be sliced
        to that point for state encoding.

        Only MELD events are decision points — WAITING, ROUND_WON,
        and COMPLETE are not decisions.
        """
        return [
            (i, event)
            for i, event in enumerate(history._memory)
            if event.player is player
            and event.event_type == EventType.MELD
        ]
