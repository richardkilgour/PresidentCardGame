#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate training, validation, and test datasets for cloning a card player.

Usage:
    python generate_clone_data.py <experiment_name> [--force]

Reads config from:
    training/experiments/<experiment_name>/config.json

The target player is specified in config["target_player"]["class"] (dotted module.ClassName).
Example:
    {
      "target_player": {
        "class": "president.players.PlayerSimple.PlayerSimple",
        "name": "simple"
      }
    }

Writes data to:
    training/experiments/<experiment_name>/data/
        X_train.npy  Y_train.npy
        X_val.npy    Y_val.npy
        X_test.npy   Y_test.npy

Skips generation if all six files already exist (unless --force is passed).

Training:   exhaustive (hand, target) pairs for N hands
Validation: random     (hand, target) pairs for M hands
Test:       random     (hand, target) pairs for M hands

Hands in all three sets are disjoint.

Player state setup
------------------
query_player() accepts a PlayerSetupFn — a callable (player, hand, target) -> None
that configures the player's internal state before play() is called.

simple_player_setup (the default) sets _hand and target_meld, which is sufficient
for any rule-based player.

For players whose play() reads additional state (e.g. full game history, opponent
card counts, network hidden state), pass a custom setup_fn that populates that
state appropriately for the synthetic single-decision context.
"""
import argparse
import importlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from president.core.AbstractPlayer import AbstractPlayer
from president.core.PlayingCard import PlayingCard
from president.core.Meld import Meld
from president.core.StateEncoder import StateEncoder

ACTION_BITS = 55   # 0-53 melds, 54 = pass
MELD_BITS   = 54
JOKER_VALUE = 13

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"

DATA_FILES = [
    "X_train.npy", "Y_train.npy",
    "X_val.npy",   "Y_val.npy",
    "X_test.npy",  "Y_test.npy",
]

DEFAULT_DATA_CONFIG = {
    "train_hands": 10_000,
    "val_hands":   12_000,
    "test_hands":  12_000,
    "seed":        42,
}

# Callable that configures the target player for a given (hand, target) scenario.
# Signature: (player, hand, target_meld_or_None) -> None
PlayerSetupFn = Callable[[AbstractPlayer, list, Optional[Meld]], None]


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

def load_config(exp_dir: Path) -> dict:
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        sys.exit(f"config.json not found in {exp_dir}")
    with open(config_path) as f:
        cfg = json.load(f)

    data_cfg = cfg.get("data", {})
    for key, value in DEFAULT_DATA_CONFIG.items():
        data_cfg.setdefault(key, value)
    cfg["data"] = data_cfg

    player_cfg = cfg.get("target_player")
    if player_cfg is None or "class" not in player_cfg:
        sys.exit(
            "config.json must contain a 'target_player' section with a 'class' key.\n"
            "Example: {\"target_player\": {\"class\": \"president.players.PlayerSimple.PlayerSimple\"}}"
        )
    player_cfg.setdefault("name", "player")
    cfg["target_player"] = player_cfg

    return cfg


# ─────────────────────────────────────────────
# Target player
# ─────────────────────────────────────────────

def simple_player_setup(player: AbstractPlayer, hand: list, target) -> None:
    """
    Default state-setup for players that only need _hand and target_meld.
    Suitable for rule-based players.
    """
    player._hand = sorted(hand, key=lambda c: c.get_index())
    player.target_meld = target


def load_player(player_cfg: dict) -> AbstractPlayer:
    """
    Instantiate the target player from config.

    Required config key:
        class  – dotted module.ClassName
                 e.g. "president.players.PlayerSimple.PlayerSimple"
    Optional config key:
        name   – player name string (default "player")
    """
    dotted = player_cfg["class"]
    module_path, class_name = dotted.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except (ModuleNotFoundError, AttributeError) as exc:
        sys.exit(f"Cannot load target player class '{dotted}': {exc}")
    return cls(player_cfg.get("name", "player"))


def query_player(
    player: AbstractPlayer,
    hand: list,
    target,
    setup_fn: PlayerSetupFn = simple_player_setup,
) -> np.ndarray:
    """
    Ask the target player what it would play given hand and target.

    setup_fn is called first to configure the player's internal state.
    For simple rule-based players this means setting _hand and target_meld.
    For more complex players (e.g. those that condition on game history or
    carry recurrent state), provide a custom setup_fn.
    """
    setup_fn(player, hand, target)
    return encode_action(player.play())


# ─────────────────────────────────────────────
# Hand generation
# ─────────────────────────────────────────────

def random_hand(hand_size: int) -> list:
    deck = [PlayingCard(i) for i in range(54)]
    hand = random.sample(deck, hand_size)
    hand.sort(key=lambda c: c.get_index())
    return hand


def random_hand_size() -> int:
    """Weighted toward mid-game hand sizes."""
    sizes   = list(range(1, 14))
    weights = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 2, 2, 1]
    return random.choices(sizes, weights=weights)[0]


# ─────────────────────────────────────────────
# Target enumeration
# ─────────────────────────────────────────────

def valid_targets_for_hand(hand: list) -> list:
    """
    All melds that could legally be a target given this hand, plus None
    (opening lead). A target meld is excluded only if the player holds
    all copies of that value, making it impossible as an opponent's play.
    """
    counts  = Counter(c.get_value() for c in hand)
    targets = [None]  # opening lead — always valid

    for value in range(14):
        max_count = 2 if value == JOKER_VALUE else 4
        held = counts.get(value, 0)
        available_to_opponent = max_count - held
        for size in range(1, available_to_opponent + 1):
            targets.append(_build_meld(value, size))

    return targets


def random_target_for_hand(hand: list):
    """Sample one target uniformly from the valid set."""
    return random.choice(valid_targets_for_hand(hand))


def _build_meld(value: int, size: int) -> Meld:
    """Build a Meld of `size` cards all of `value`."""
    meld = None
    for suit in range(size):
        card = PlayingCard(52 + suit) if value == JOKER_VALUE \
               else PlayingCard(value * 4 + suit)
        meld = Meld(card, meld) if meld else Meld(card)
    return meld


# ─────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────

def encode_example(hand: list, target) -> tuple[np.ndarray, np.ndarray]:
    hand_enc   = StateEncoder._encode_hand(hand)
    target_enc = StateEncoder.encode_meld(target) if target is not None \
                 else np.zeros(MELD_BITS, dtype=np.int8)
    return hand_enc, target_enc


def encode_action(meld) -> np.ndarray:
    """55-class one-hot: 0-53 are melds, 54 is pass."""
    vec = np.zeros(ACTION_BITS, dtype=np.int8)
    if meld is None or not meld.cards:
        vec[54] = 1
    else:
        vec[:54] = StateEncoder.encode_meld(meld)
    return vec


# ─────────────────────────────────────────────
# Dataset builders
# ─────────────────────────────────────────────

def build_training_set(
    n_hands: int,
    player: AbstractPlayer,
    setup_fn: PlayerSetupFn = simple_player_setup,
) -> tuple:
    """
    Exhaustive: every valid (hand, target) pair for each hand.
    Returns X (N, 108), Y (N, 55).
    """
    X, Y = [], []
    for _ in range(n_hands):
        hand = random_hand(random_hand_size())
        for target in valid_targets_for_hand(hand):
            hand_enc, target_enc = encode_example(hand, target)
            action_enc = query_player(player, hand, target, setup_fn)
            X.append(np.concatenate([hand_enc, target_enc]))
            Y.append(action_enc)
    return np.array(X, dtype=np.int8), np.array(Y, dtype=np.int8)


def build_random_set(
    n_hands: int,
    player: AbstractPlayer,
    setup_fn: PlayerSetupFn = simple_player_setup,
) -> tuple:
    """
    Random: one random target per hand.
    Used for validation and test sets.
    Returns X (N, 108), Y (N, 55).
    """
    X, Y = [], []
    for _ in range(n_hands):
        hand   = random_hand(random_hand_size())
        target = random_target_for_hand(hand)
        hand_enc, target_enc = encode_example(hand, target)
        action_enc = query_player(player, hand, target, setup_fn)
        X.append(np.concatenate([hand_enc, target_enc]))
        Y.append(action_enc)
    return np.array(X, dtype=np.int8), np.array(Y, dtype=np.int8)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate clone training data for a card player."
    )
    parser.add_argument("experiment", help="Experiment name under training/experiments/")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate data even if it already exists")
    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.experiment
    if not exp_dir.is_dir():
        sys.exit(f"Experiment directory not found: {exp_dir}")

    data_dir = exp_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # ── Skip if data already complete ─────────
    existing = [f for f in DATA_FILES if (data_dir / f).exists()]
    if len(existing) == len(DATA_FILES) and not args.force:
        print(f"Data already exists in {data_dir} — skipping generation.")
        print("Pass --force to regenerate.")
        return

    if args.force and existing:
        print(f"--force passed; regenerating {len(existing)} existing file(s).")

    # ── Config ────────────────────────────────
    cfg        = load_config(exp_dir)
    data_cfg   = cfg["data"]
    player_cfg = cfg["target_player"]

    print(f"Experiment    : {args.experiment}")
    print(f"Target player : {player_cfg['class']} (name={player_cfg['name']!r})")
    print(f"Data config   : {json.dumps(data_cfg, indent=2)}\n")

    random.seed(data_cfg["seed"])
    np.random.seed(data_cfg["seed"])

    # ── Target player ─────────────────────────
    player   = load_player(player_cfg)
    setup_fn = simple_player_setup
    # For players that need richer state, replace setup_fn here or derive it
    # from player_cfg (e.g. a "setup" key pointing to a custom callable).

    # ── Generate ──────────────────────────────
    print(f"Generating training set   ({data_cfg['train_hands']:,} hands, exhaustive)...")
    X_train, Y_train = build_training_set(data_cfg["train_hands"], player, setup_fn)

    print(f"Generating validation set ({data_cfg['val_hands']:,} hands, random)...")
    X_val, Y_val = build_random_set(data_cfg["val_hands"], player, setup_fn)

    print(f"Generating test set       ({data_cfg['test_hands']:,} hands, random)...")
    X_test, Y_test = build_random_set(data_cfg["test_hands"], player, setup_fn)

    # ── Report ────────────────────────────────
    print(f"\nTraining:   {len(X_train):>7,} examples")
    print(f"Validation: {len(X_val):>7,} examples")
    print(f"Test:       {len(X_test):>7,} examples")

    # ── Save ──────────────────────────────────
    np.save(data_dir / "X_train.npy", X_train)
    np.save(data_dir / "Y_train.npy", Y_train)
    np.save(data_dir / "X_val.npy",   X_val)
    np.save(data_dir / "Y_val.npy",   Y_val)
    np.save(data_dir / "X_test.npy",  X_test)
    np.save(data_dir / "Y_test.npy",  Y_test)

    print(f"\nSaved to {data_dir}")


if __name__ == "__main__":
    main()
