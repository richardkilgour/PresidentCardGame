#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CloneDataCollector — CardGameListener that records training data from live games.

Register an instance as a GameMaster listener before calling start().
After the game finishes, call flat_decision_points() to get all recorded
decisions, or access episode_records directly for sequence-level data.

Meld context layout (relative to the tracked player's seat s):
    melds[0]  meld_right    = current_melds[(s-1) % 4]  — most recent play before us
    melds[1]  meld_opposite = current_melds[(s-2) % 4]
    melds[2]  meld_left     = current_melds[(s-3) % 4]
    melds[3]  meld_self     = current_melds[s]           — our own last play this hand
"""
from __future__ import annotations

from typing import Optional

from president.core.CardGameListener import CardGameListener
from president.core.Meld import Meld
from president.training.data.records import DecisionPoint, EpisodeRecord


class CloneDataCollector(CardGameListener):
    """
    Listens to a live game and records every decision made by any player
    whose type matches target_cls.

    If all four seats are the same class, all four players are recorded
    (4x decisions per episode). If two seats match, 2x, etc.

    Usage:
        collector = CloneDataCollector(PlayerSplitter)
        gm = GameMaster(...)
        gm.add_listener(collector)          # before add_player so seats are tracked
        gm.add_player(target_player)
        for opp in opponents:
            gm.add_player(opp)
        gm.start(number_of_rounds=n)
        while not gm.step(): pass

        data = collector.flat_decision_points()
    """

    def __init__(self, target_cls: type) -> None:
        super().__init__()
        self.target_cls = target_cls
        # current_melds[seat] = last Meld played by that seat this hand, or None
        self._current_melds: list[Optional[Meld]] = [None, None, None, None]
        # snapshot taken when notify_player_turn fires for the target
        self._pending_hand:                    Optional[list] = None
        self._pending_melds:                   Optional[list] = None
        self._pending_play_history_snapshot:   Optional[list] = None
        # output
        self.episode_records: list[EpisodeRecord] = []
        self._current_episode: EpisodeRecord = EpisodeRecord()

    # ------------------------------------------------------------------ #
    # Hand lifecycle                                                        #
    # ------------------------------------------------------------------ #

    def notify_hand_start(self) -> None:
        super().notify_hand_start()
        self._reset_melds()

    def notify_hand_won(self, winner) -> None:
        super().notify_hand_won(winner)
        self._reset_melds()

    def _reset_melds(self) -> None:
        self._current_melds = [None, None, None, None]

    # ------------------------------------------------------------------ #
    # Episode lifecycle                                                     #
    # ------------------------------------------------------------------ #

    def notify_episode_end(self, final_ranks: list, starting_ranks: list) -> None:
        """Seal the current episode record and start a fresh one."""
        self.episode_records.append(self._current_episode)
        self._current_episode = EpisodeRecord()

    # ------------------------------------------------------------------ #
    # Turn tracking                                                         #
    # ------------------------------------------------------------------ #

    def notify_player_turn(self, player) -> None:
        """Snapshot state just before a target-class player acts."""
        if not isinstance(player, self.target_cls):
            return
        seat = self.players.index(player)
        s = seat
        self._pending_hand  = list(player._hand)
        self._pending_melds = [
            self._current_melds[(s - 1) % 4],   # meld_right
            self._current_melds[(s - 2) % 4],   # meld_opposite
            self._current_melds[(s - 3) % 4],   # meld_left
            self._current_melds[s],              # meld_self
        ]
        self._pending_play_history_snapshot = list(self.memory._memory)

    def notify_play(self, player, meld: Meld) -> None:
        """Update table state; record if this is the target's action."""
        super().notify_play(player, meld)
        seat = self.players.index(player)
        self._current_melds[seat] = meld
        if isinstance(player, self.target_cls):
            self._commit(meld)

    def notify_pass(self, player) -> None:
        """Passes don't change current_melds; record if target passed."""
        super().notify_pass(player)
        if isinstance(player, self.target_cls):
            self._commit(Meld())   # empty Meld = pass

    # ------------------------------------------------------------------ #
    # Internal                                                              #
    # ------------------------------------------------------------------ #

    def _commit(self, action: Meld) -> None:
        """Pair the pending snapshot with the observed action."""
        if self._pending_hand is None:
            return   # no snapshot — target was already finished (shouldn't happen)
        dp = DecisionPoint(
            hand=self._pending_hand,
            melds=self._pending_melds,
            action=action,
            play_history_snapshot=self._pending_play_history_snapshot or [],
        )
        self._current_episode.decision_points.append(dp)
        self._pending_hand                  = None
        self._pending_melds                 = None
        self._pending_play_history_snapshot = None

    # ------------------------------------------------------------------ #
    # Output                                                                #
    # ------------------------------------------------------------------ #

    def flat_decision_points(self) -> list[DecisionPoint]:
        """All recorded decisions across every completed episode."""
        return [
            dp
            for ep in self.episode_records
            for dp in ep.decision_points
        ]
