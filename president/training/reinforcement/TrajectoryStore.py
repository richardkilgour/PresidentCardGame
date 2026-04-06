#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TrajectoryStore appends, loads, and filters Trajectory records
from per-policy .jsonl files.

File format: one JSON record per line, one file per policy.
Files are append-only and never overwritten, so data accumulates
safely across sessions.

Usage:
    # Save
    TrajectoryStore.append(trajectory)

    # Load all trajectories for a policy
    trajectories = TrajectoryStore.load("PlayerSplitter")

    # Load and filter
    trajectories = TrajectoryStore.load(
        "PlayerSplitter",
        final_rank=0,           # President only
        min_turns=5,            # at least 5 decision points
        opponent_policy="PlayerHolder",  # played against Holder
    )

    # Summary stats
    TrajectoryStore.summary("PlayerSplitter")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from president.training.reinforcement.Trajectory import Trajectory


DEFAULT_DATA_DIR = Path("data/trajectories")


class TrajectoryStore:

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
        """
        Args:
            data_dir: Directory where .jsonl files are stored.
                      Created if it does not exist.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    def append(self, trajectory: Trajectory) -> None:
        """
        Append a trajectory to the policy's .jsonl file.
        Creates the file if it does not exist.

        Args:
            trajectory: The Trajectory to save.
        """
        path = self._path_for(trajectory.metadata.policy)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trajectory.to_dict()) + "\n")
        logging.info(
            f"TrajectoryStore: appended {trajectory} to {path.name}"
        )

    def append_all(self, trajectories: list[Trajectory]) -> None:
        """Append multiple trajectories, grouped by policy for efficiency."""
        by_policy: dict[str, list[Trajectory]] = {}
        for t in trajectories:
            by_policy.setdefault(t.metadata.policy, []).append(t)
        for policy, group in by_policy.items():
            path = self._path_for(policy)
            with path.open("a", encoding="utf-8") as f:
                for t in group:
                    f.write(json.dumps(t.to_dict()) + "\n")
            logging.info(
                f"TrajectoryStore: appended {len(group)} trajectories "
                f"to {path.name}"
            )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load(self, policy: str,
             final_rank: int | None = None,
             min_turns: int | None = None,
             opponent_policy: str | None = None,
             max_trajectories: int | None = None) -> list[Trajectory]:
        """
        Load trajectories for a policy, with optional filtering.

        Args:
            policy:            Policy name e.g. "PlayerSplitter".
            final_rank:        Filter to episodes where player finished
                               at this rank (0=President .. 3=Scumbag).
            min_turns:         Filter to episodes with at least this many
                               decision points.
            opponent_policy:   Filter to episodes containing at least one
                               opponent with this policy.
            max_trajectories:  Cap on number of trajectories to load.
                               Loads most recent if capped.

        Returns:
            List of Trajectory objects matching all filters.
        """
        path = self._path_for(policy)
        if not path.exists():
            logging.warning(
                f"TrajectoryStore: no data file for policy '{policy}' "
                f"at {path}"
            )
            return []

        trajectories = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = Trajectory.from_dict(json.loads(line))
                except Exception as e:
                    logging.warning(
                        f"TrajectoryStore: skipping malformed record: {e}"
                    )
                    continue
                if self._matches(t, final_rank, min_turns, opponent_policy):
                    trajectories.append(t)

        # Cap to most recent N if requested
        if max_trajectories and len(trajectories) > max_trajectories:
            trajectories = trajectories[-max_trajectories:]

        logging.info(
            f"TrajectoryStore: loaded {len(trajectories)} trajectories "
            f"for '{policy}'"
        )
        return trajectories

    def load_supervised(self, policy: str,
                        **filters) -> tuple[np.ndarray, np.ndarray]:
        """
        Load all (state, action) pairs for supervised training.

        Args:
            policy:   Policy name.
            **filters: Passed to load().

        Returns:
            states:  (N, 341) array
            actions: (N, 54)  array
        """
        trajectories = self.load(policy, **filters)
        if not trajectories:
            raise RuntimeError(
                f"TrajectoryStore: no trajectories found for '{policy}' "
                f"with the given filters."
            )
        all_states  = np.vstack([t.states  for t in trajectories])
        all_actions = np.vstack([t.actions for t in trajectories])
        logging.info(
            f"TrajectoryStore: {len(all_states)} supervised samples "
            f"for '{policy}'"
        )
        return all_states, all_actions

    def load_rl(self, policy: str,
                **filters) -> tuple[np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray, np.ndarray]:
        """
        Load full RL trajectories as stacked numpy arrays.

        Returns:
            states, actions, rewards, next_states, dones
        """
        trajectories = self.load(policy, **filters)
        if not trajectories:
            raise RuntimeError(
                f"TrajectoryStore: no trajectories found for '{policy}'."
            )
        parts = [t.to_rl() for t in trajectories]
        return tuple(np.vstack(a) for a in zip(*parts))

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self, policy: str) -> str:
        """
        Print a human-readable summary of stored trajectories for a policy.

        Returns:
            Formatted summary string.
        """
        trajectories = self.load(policy)
        if not trajectories:
            return f"No trajectories found for '{policy}'."

        rank_names  = ["President", "Vice-President", "Citizen", "Scumbag"]
        rank_counts = [0, 0, 0, 0]
        total_turns = 0
        opponents_seen: dict[str, int] = {}

        for t in trajectories:
            rank_counts[t.metadata.final_rank] += 1
            total_turns += len(t.states)
            for o in t.metadata.opponents:
                opponents_seen[o.policy] = \
                    opponents_seen.get(o.policy, 0) + 1

        n = len(trajectories)
        lines = [
            f"=== Trajectory summary: {policy} ===",
            f"Episodes:      {n}",
            f"Total turns:   {total_turns}",
            f"Avg turns/ep:  {total_turns / n:.1f}",
            f"\nFinishing positions:",
        ]
        for i, name in enumerate(rank_names):
            pct = 100 * rank_counts[i] / n
            lines.append(f"  {name:18} {rank_counts[i]:4d}  ({pct:.1f}%)")

        lines.append(f"\nOpponents faced:")
        for policy_name, count in sorted(
            opponents_seen.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {policy_name:20} {count:4d} times")

        result = "\n".join(lines)
        print(result)
        return result

    def available_policies(self) -> list[str]:
        """Return all policy names that have data files."""
        return [p.stem for p in self.data_dir.glob("*.jsonl")]

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _path_for(self, policy: str) -> Path:
        return self.data_dir / f"{policy}.jsonl"

    @staticmethod
    def _matches(trajectory: Trajectory,
                 final_rank: int | None,
                 min_turns: int | None,
                 opponent_policy: str | None) -> bool:
        if final_rank is not None and \
                trajectory.metadata.final_rank != final_rank:
            return False
        if min_turns is not None and \
                len(trajectory.states) < min_turns:
            return False
        if opponent_policy is not None:
            policies = {o.policy for o in trajectory.metadata.opponents}
            if opponent_policy not in policies:
                return False
        return True
