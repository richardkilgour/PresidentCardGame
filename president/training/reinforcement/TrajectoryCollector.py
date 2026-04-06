#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TrajectoryCollector is a CardGameListener that builds and saves one Trajectory
per player at the end of each episode.

Add it to a GameMaster as an extra listener when trajectory collection is
needed. Players themselves have no dependency on training infrastructure.

Usage:
    collector = TrajectoryCollector()               # default data dir
    collector = TrajectoryCollector("data/run_01")  # custom data dir

    gm = GameMaster(...)
    gm.add_listener(collector)   # register before players join
    gm.add_player(...)
    gm.start(number_of_rounds=1000)
    while not gm.step():
        pass
"""
from __future__ import annotations

import logging

from president.core.CardGameListener import CardGameListener
from president.training.reinforcement.EpisodeEncoder import EpisodeEncoder
from president.training.reinforcement.TrajectoryStore import TrajectoryStore

logger = logging.getLogger(__name__)


class TrajectoryCollector(CardGameListener):
    """
    Listener that builds and saves trajectories for all seated players
    at the end of every episode.

    Args:
        data_dir: Directory where .jsonl trajectory files are written.
                  Passed through to TrajectoryStore. Uses the store's
                  default if not specified.
        players_to_track: Optional list of players to record. When None
                          (default) every seated player is recorded.
    """

    def __init__(self, data_dir=None, players_to_track=None) -> None:
        super().__init__()
        self._encoder = EpisodeEncoder()
        self._store   = TrajectoryStore(data_dir) if data_dir \
                        else TrajectoryStore()
        self._players_to_track = players_to_track  # None → record all

    def notify_episode_end(self, final_ranks: list,
                           starting_ranks: list) -> None:
        """
        Called once per episode. Encodes and saves a trajectory for each
        tracked player. Players with no decision points (e.g. a waiting
        player in a short episode) are silently skipped.
        """
        for player in self._players:
            if player is None:
                continue
            if self._players_to_track is not None \
                    and player not in self._players_to_track:
                continue
            try:
                opponents  = self.opponents_clockwise(player)
                trajectory = self._encoder.encode(player, opponents)
                self._store.append(trajectory)
            except RuntimeError as e:
                logger.warning(
                    f"TrajectoryCollector: skipping {player.name}: {e}"
                )
