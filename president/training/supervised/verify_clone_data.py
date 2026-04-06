#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify clone training data against the target player.

For each sample (X[i], Y[i]), decodes the hand and target meld from X[i],
queries a fresh instance of the target player, and checks that its action
exactly matches Y[i].  Since rule-based players are deterministic, any
mismatch indicates a data-generation bug.

Usage:
    python verify_clone_data.py <experiment_name> [options]

Options:
    --split {train,val,test}   Which dataset to check (default: train)
    --samples N                Check N randomly chosen samples
    --exhaustive               Check every sample (default if --samples omitted)
"""
import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np

from president.core.PlayingCard import PlayingCard
from president.core.Meld import Meld
from president.training.data.StateEncoder import StateEncoder, MELD_BITS, HAND_BITS, JOKER_OFFSET, MAX_REGULAR_VALUE

EXPERIMENTS_DIR = (Path(__file__).resolve().parent.parent / "experiments").resolve()


# ─────────────────────────────────────────────
# Decode helpers
# ─────────────────────────────────────────────

def decode_hand(bits: np.ndarray) -> list:
    """
    Reconstruct a hand from a 54-bit cumulative-count encoding.

    Suit information is not preserved by the encoding; cards are
    reconstructed with suits 0, 1, 2, 3 in order — sufficient for
    any rule-based player that reasons only about card values.
    """
    hand = []
    for value in range(MAX_REGULAR_VALUE):          # 0-12, regular cards
        count = int(np.sum(bits[value * 4: value * 4 + 4]))
        for suit in range(count):
            hand.append(PlayingCard(value * 4 + suit))
    joker_count = int(np.sum(bits[JOKER_OFFSET: JOKER_OFFSET + 2]))
    for j in range(joker_count):
        hand.append(PlayingCard(52 + j))
    return hand


def decode_target(bits: np.ndarray):
    """
    Decode a 54-bit one-hot meld encoding to a Meld, or None for all-zeros
    (opening lead / no current target).
    """
    if not np.any(bits):
        return None
    return StateEncoder.decode_meld(bits)


# ─────────────────────────────────────────────
# Config / player loading
# ─────────────────────────────────────────────

def _load_class(dotted: str):
    module_path, class_name = dotted.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ModuleNotFoundError, AttributeError) as exc:
        sys.exit(f"Cannot load class '{dotted}': {exc}")


def load_config(exp_dir: Path) -> dict:
    path = exp_dir / "config.json"
    if not path.exists():
        sys.exit(f"config.json not found in {exp_dir}")
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────

def verify_sample(x: np.ndarray, y: np.ndarray, player, encode_action_fn) -> bool:
    """
    Decode one sample, query the player, compare to label.

    The hand occupies x[0:54].  The other players' melds (meld_right for multi-meld
    models, the only meld for single-meld models) occupies x[54:108] x[109:162].
    Any further slots in x are context the network uses but the rule-based
    player does not need.

    Returns True if the player's action matches the label exactly.
    """
    hand   = decode_hand(x[:HAND_BITS])
    meld_right    = decode_target(x[HAND_BITS:2*HAND_BITS])
    meld_opposite = decode_target(x[2*HAND_BITS:3*HAND_BITS])
    meld_left     = decode_target(x[3*HAND_BITS:4*HAND_BITS])
    melds_opponent = [meld_right, meld_opposite, meld_left]
    target = None
    for m in melds_opponent:
        if m is not None and m.cards and (target is None or m > target):
            target = m

    player._hand = sorted(hand, key=lambda c: c.get_index())
    player.target_meld = target

    action     = player.play()
    encoded    = encode_action_fn(action)
    return np.array_equal(encoded, y)


def run_verification(
    X: np.ndarray,
    Y: np.ndarray,
    player,
    encode_action_fn,
    indices,
) -> tuple[int, int, list[int]]:
    """
    Verify a subset of samples.

    Returns (n_pass, n_fail, failing_indices).
    """
    n_pass, n_fail = 0, []
    for i in indices:
        if verify_sample(X[i], Y[i], player, encode_action_fn):
            n_pass += 1
        else:
            n_fail.append(i)
    return n_pass, len(n_fail), n_fail


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify clone training data against the target player."
    )
    parser.add_argument("experiment",
                        help="Experiment name under training/experiments/")
    parser.add_argument("--split", choices=["train", "val", "test"],
                        default="train",
                        help="Dataset split to verify (default: train)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--samples", type=int, metavar="N",
                       help="Check N randomly chosen samples")
    group.add_argument("--exhaustive", action="store_true",
                       help="Check every sample (default)")
    args = parser.parse_args()

    exp_dir  = (EXPERIMENTS_DIR / args.experiment).resolve()
    data_dir = exp_dir / "data"

    if not exp_dir.is_dir():
        sys.exit(f"Experiment directory not found: {exp_dir}")

    X_path = data_dir / f"X_{args.split}.npy"
    Y_path = data_dir / f"Y_{args.split}.npy"
    if not X_path.exists() or not Y_path.exists():
        sys.exit(f"Data files not found in {data_dir}. Run generate_clone_data.py first.")

    cfg        = load_config(exp_dir)
    player_cfg = cfg["target_player"]
    player_cls = _load_class(player_cfg["class"])
    player     = player_cls(player_cfg.get("name", "player"))

    X = np.load(X_path)
    Y = np.load(Y_path)
    n_total = len(X)

    # Inline encode_action (mirrors generate_clone_data.py)
    def encode_action(meld) -> np.ndarray:
        vec = np.zeros(55, dtype=np.int8)
        if meld is None or not meld.cards:
            vec[54] = 1
        else:
            vec[:54] = StateEncoder.encode_meld(meld)
        return vec

    # Choose indices
    if args.samples:
        indices = random.sample(range(n_total), min(args.samples, n_total))
        mode_str = f"{len(indices):,} random samples"
    else:
        indices = list(range(n_total))
        mode_str = f"all {n_total:,} samples (exhaustive)"

    print(f"Experiment : {args.experiment}")
    print(f"Split      : {args.split}")
    print(f"Player     : {player_cfg['class']}")
    print(f"Mode       : {mode_str}")
    print()

    n_pass, n_fail, failing = run_verification(X, Y, player, encode_action, indices)

    print(f"Passed : {n_pass:,} / {len(indices):,}")
    print(f"Failed : {n_fail:,} / {len(indices):,}")

    if failing:
        print(f"\nFirst failing indices: {failing[:20]}")
        sys.exit(1)
    else:
        print("\nAll samples verified successfully.")


if __name__ == "__main__":
    main()
