#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Display a trajectory .jsonl file in human-readable format.

Usage:
    python debug/display_trajectory.py data/trajectories/PlayerSplitter.jsonl
    python debug/display_trajectory.py data/trajectories/PlayerSplitter.jsonl --episode 3
    python debug/display_trajectory.py data/trajectories/PlayerSplitter.jsonl --summary
    python debug/display_trajectory.py data/trajectories/PlayerSplitter.jsonl --rank 0
    python debug/display_trajectory.py data/trajectories/PlayerSplitter.jsonl --last 5
    python debug/display_trajectory.py data/trajectories/PlayerSplitter.jsonl --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from president.training.reinforcement.Trajectory import Trajectory, RANK_NAMES

sys.path.insert(0, str(Path(__file__).parent.parent))

from president.core.StateEncoder import (
    StateEncoder,
    BLOCK_0_SIZE, BLOCK_N_SIZE,
    STATE_BITS, HAND_SIZE_BITS, MELD_BITS, HAND_BITS,
    IDX_WAITING, IDX_HAS_PLAYED, IDX_WON_ROUND,
    JOKER_OFFSET,
)

STATE_NAMES  = ["waiting", "has_played", "won_round"]
RANK_SYMBOLS = ["👑", "🥈", "👤", "💩"]


# ─────────────────────────────────────────────
# Decoding helpers
# ─────────────────────────────────────────────

def decode_state_bits(bits: np.ndarray) -> str:
    for i, name in enumerate(STATE_NAMES):
        if bits[i]:
            return name
    return "passed/finished"


def decode_hand(bits: np.ndarray) -> str:
    rank_names = ["3","4","5","6","7","8","9","10","J","Q","K","A","2","Joker"]
    cards = []
    for value in range(13):
        offset    = value * 4
        max_count = 2 if value == 13 else 4
        count     = int(np.sum(bits[offset:offset + max_count]))
        if count:
            cards.append(f"{count}x{rank_names[value]}")
    joker_count = int(np.sum(bits[JOKER_OFFSET:JOKER_OFFSET + 2]))
    if joker_count:
        cards.append(f"{joker_count}xJoker")
    return "  ".join(cards) if cards else "(empty)"


def decode_hand_size(bits: np.ndarray) -> int:
    return int(np.sum(bits))


def decode_meld_bits(bits: np.ndarray) -> str:
    meld = StateEncoder.decode_meld(bits)
    if not meld.cards:
        return "<pass>"
    rank_names = ["3","4","5","6","7","8","9","10","J","Q","K","A","2","Joker"]
    return f"{len(meld)}x{rank_names[meld.cards[0].get_value()]}"


def block_player_name(block_num: int, metadata) -> str:
    """
    Map history block number to player name.
    Turn order is fixed and clockwise:
      t-1 → opponents[2]  (anticlockwise neighbour)
      t-2 → opponents[1]  (opposite)
      t-3 → opponents[0]  (clockwise neighbour)
      t-4 → self
    """
    mapping = {
        1: metadata.opponents[2].name if len(metadata.opponents) > 2 else "?",
        2: metadata.opponents[1].name if len(metadata.opponents) > 1 else "?",
        3: metadata.opponents[0].name if len(metadata.opponents) > 0 else "?",
        4: metadata.player_name,
    }
    return mapping.get(block_num, "?")


def decode_history_line(state: np.ndarray, metadata) -> str:
    """
    Decode blocks 1-4 into a single concise history line.
    e.g. snAkbar (9 cards) played 1x4;  Richard (11 cards) passed
    """
    parts = []
    for block_num in range(1, 5):
        offset      = BLOCK_0_SIZE + (block_num - 1) * BLOCK_N_SIZE
        b_state     = state[offset:offset + STATE_BITS]
        b_hand_size = state[offset + STATE_BITS:
                            offset + STATE_BITS + HAND_SIZE_BITS]
        b_meld      = state[offset + STATE_BITS + HAND_SIZE_BITS:
                            offset + BLOCK_N_SIZE]

        # Skip zero-padded empty blocks
        if np.sum(b_state) == 0 and np.sum(b_meld) == 0 \
                and decode_hand_size(b_hand_size) == 0:
            continue

        name   = block_player_name(block_num, metadata)
        size   = decode_hand_size(b_hand_size)
        meld   = decode_meld_bits(b_meld)
        state_ = decode_state_bits(b_state)

        if size == 0:
            parts.append(f"{name} (0 cards) finished")
        elif meld == "<pass>":
            parts.append(f"{name} ({size} cards) passed")
        else:
            parts.append(f"{name} ({size} cards) played {meld}")

    return ";  ".join(parts) if parts else ""


# ─────────────────────────────────────────────
# Trajectory display
# ─────────────────────────────────────────────

def display_trajectory(t: Trajectory, verbose: bool = False) -> str:
    lines = []
    m = t.metadata

    # --- Header ---
    lines.append(f"\n{'═' * 70}")
    lines.append(
        f"  {m.player_name} ({m.policy})  |  "
        f"{m.starting_rank_name} → "
        f"{RANK_SYMBOLS[m.final_rank]} {m.final_rank_name}  |  "
        f"{len(t.states)} turns  |  "
        f"{m.timestamp[:19]}"
    )
    lines.append(f"  Episode: {m.episode_id[:8]}")

    # --- Opponents ---
    lines.append(f"{'─' * 70}")
    lines.append("  Opponents:")
    for o in m.opponents:
        was = o.starting_rank_name
        now = f"{RANK_SYMBOLS[o.final_rank]} {o.final_rank_name}"
        lines.append(f"    {o.name:15} ({o.policy:20})  "
                     f"Was: {was:18}  Now: {now}")

    # --- Turns ---
    lines.append(f"{'─' * 70}")
    for i, (state, action, done) in enumerate(
        zip(t.states, t.actions, t.done)
    ):
        state      = np.array(state)
        action_str = decode_meld_bits(action)
        hand_bits  = state[STATE_BITS:STATE_BITS + HAND_BITS]
        state_str  = decode_state_bits(state[:STATE_BITS])
        history    = decode_history_line(state, m)

        if verbose:
            # Full state breakdown
            lines.append(
                f"  Turn {i+1:3d}  "
                f"state={state_str:15}  "
                f"hand={decode_hand(hand_bits)}"
            )
            lines.append(
                f"           played={action_str:15}"
                f"{'  ← DONE' if done else ''}"
            )
        else:
            lines.append(
                f"  Turn {i+1:3d}  "
                f"state={state_str:12}  "
                f"hand={decode_hand(hand_bits):35}  "
                f"played={action_str:15}"
                f"{'  DONE' if done else ''}"
            )

        if history:
            lines.append(f"           {history}")

    lines.append(f"{'═' * 70}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────

def display_summary(trajectories: list[Trajectory]) -> str:
    if not trajectories:
        return "No trajectories found."

    rank_counts     = [0, 0, 0, 0]
    total_turns     = 0
    opponents_seen: dict[str, int] = {}

    for t in trajectories:
        rank_counts[t.metadata.final_rank] += 1
        total_turns += len(t.states)
        for o in t.metadata.opponents:
            opponents_seen[o.policy] = opponents_seen.get(o.policy, 0) + 1

    n = len(trajectories)
    lines = [
        f"\n{'═' * 70}",
        f"  Policy:        {trajectories[0].metadata.policy}",
        f"  Episodes:      {n}",
        f"  Total turns:   {total_turns}",
        f"  Avg turns/ep:  {total_turns / n:.1f}",
        f"{'─' * 70}",
        f"  Finishing positions:",
    ]
    for i, name in enumerate(RANK_NAMES):
        pct = 100 * rank_counts[i] / n
        bar = "█" * int(pct / 2)
        lines.append(
            f"    {RANK_SYMBOLS[i]} {name:18} "
            f"{rank_counts[i]:4d}  ({pct:5.1f}%)  {bar}"
        )
    lines.append(f"{'─' * 70}")
    lines.append(f"  Opponents faced:")
    for policy, count in sorted(
        opponents_seen.items(), key=lambda x: -x[1]
    ):
        lines.append(f"    {policy:25} {count:4d} times")
    lines.append(f"{'═' * 70}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────

def load_trajectories(path: Path,
                      rank_filter: int | None = None) -> list[Trajectory]:
    trajectories = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = Trajectory.from_dict(json.loads(line))
                if rank_filter is None or t.metadata.final_rank == rank_filter:
                    trajectories.append(t)
            except Exception as e:
                print(f"Warning: skipping malformed record: {e}")
    return trajectories


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Display a trajectory .jsonl file in human-readable format."
    )
    parser.add_argument("file", type=str,
                        help="Path to the .jsonl trajectory file.")
    parser.add_argument("--episode", type=int, default=None, metavar="N",
                        help="Display only episode N (1-indexed).")
    parser.add_argument("--summary", action="store_true",
                        help="Display aggregate summary only.")
    parser.add_argument("--rank", type=int, default=None, metavar="R",
                        help="Filter to episodes where player finished "
                             "at rank R (0-3).")
    parser.add_argument("--verbose", action="store_true",
                        help="Show full state breakdown per turn.")
    parser.add_argument("--last", type=int, default=None, metavar="N",
                        help="Display only the last N episodes.")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    trajectories = load_trajectories(path, rank_filter=args.rank)
    if not trajectories:
        print("No trajectories found matching the given filters.")
        sys.exit(0)

    print(f"Loaded {len(trajectories)} trajectories from {path.name}")

    if args.summary:
        print(display_summary(trajectories))
        return

    if args.last:
        trajectories = trajectories[-args.last:]

    if args.episode:
        if args.episode < 1 or args.episode > len(trajectories):
            print(f"Episode {args.episode} out of range "
                  f"(1-{len(trajectories)}).")
            sys.exit(1)
        print(display_trajectory(
            trajectories[args.episode - 1], verbose=args.verbose
        ))
        return

    for t in trajectories:
        print(display_trajectory(t, verbose=args.verbose))

    print(display_summary(trajectories))


if __name__ == "__main__":
    main()
