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

Usage:
    encoder = EpisodeEncoder()
    trajectory = encoder.encode(
        player=player,
        final_ranks=episode.ranks,
        starting_ranks=game_master.positions,
        opponents=opponents,
    )
    TrajectoryStore.append(trajectory)
"""
import numpy as np

from president.core.PlayHistory import EventType, PlayHistory
from president.core.StateEncoder import StateEncoder, MELD_BITS, TOTAL_BITS
from president.core.Trajectory import Trajectory, TrajectoryMetadata


class EpisodeEncoder:
    """
    Builds a Trajectory from a player's PlayHistory at episode end.

    Reads entirely from PlayHistory — no live player state is used.
    One EpisodeEncoder can be reused across many episodes and players.
    """

    def __init__(self) -> None:
        self._state_encoder = StateEncoder()

    def encode(self, player, final_ranks: list,
               starting_ranks: list, opponents: list) -> Trajectory:
        """
        Encode the full episode from one player's perspective.

        Args:
            player:         The player whose trajectory to encode.
                            Only player.memory (PlayHistory) and
                            player identity are used — no live state.
            final_ranks:    All players in finishing order.
            starting_ranks: All players in starting order from the
                            previous episode. Empty if first episode.
            opponents:      Other players in clockwise order.

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
            final_ranks=final_ranks,
            starting_ranks=starting_ranks,
            opponents=opponents,
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
