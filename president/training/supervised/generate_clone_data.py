#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate training, validation, and test datasets for cloning a card player.

Usage:
    python generate_clone_data.py <experiment_name> [--force]

Reads config from:
    training/experiments/<experiment_name>/config.json

The target player is specified in config["target_player"]["class"] (dotted module.ClassName).

Two data-generation modes, selected by the model class:

  Synthetic (model.MULTI_MELD not set):
    Exhaustive (hand, target) pairs for training; random for val/test.
    Config keys: train_hands, val_hands, test_hands, seed.

  Episode-based (model.MULTI_MELD = True):
    Full games with real opponents; CloneDataCollector records decisions.
    Config keys: train_episodes, val_episodes, test_episodes, seed.
    Requires config["opponents"] — list of {class, name} dicts (3 players).

Writes data to:
    training/experiments/<experiment_name>/data/
        X_train.npy  Y_train.npy
        X_val.npy    Y_val.npy
        X_test.npy   Y_test.npy

Episode-based mode also writes:
        episodes_train.pkl   episodes_val.pkl   episodes_test.pkl
    Each pickle contains a list[EpisodeRecord] for future RNN training.

Skips generation if all required files already exist (unless --force is passed).
"""
import argparse
import importlib
import json
import pickle
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
JOKER_VALUE = 13

EXPERIMENTS_DIR = (Path(__file__).resolve().parent.parent / "experiments").resolve()

DATA_FILES = [
    "X_train.npy", "Y_train.npy",
    "X_val.npy",   "Y_val.npy",
    "X_test.npy",  "Y_test.npy",
]

EPISODE_FILES = [
    "episodes_train.pkl",
    "episodes_val.pkl",
    "episodes_test.pkl",
]

DEFAULT_DATA_CONFIG = {
    "train_hands": 10_000,
    "val_hands":   12_000,
    "test_hands":  12_000,
    "seed":        42,
}

DEFAULT_EPISODE_DATA_CONFIG = {
    "train_episodes": 5_000,
    "val_episodes":   1_000,
    "test_episodes":  1_000,
    "seed":           42,
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

    player_cfg = cfg.get("target_player")
    if player_cfg is None or "class" not in player_cfg:
        sys.exit(
            "config.json must contain a 'target_player' section with a 'class' key.\n"
            "Example: {\"target_player\": {\"class\": \"president.players.PlayerSimple.PlayerSimple\"}}"
        )
    player_cfg.setdefault("name", "player")
    cfg["target_player"] = player_cfg

    model_cfg = cfg.get("model")
    if model_cfg is None or "class" not in model_cfg:
        sys.exit(
            "config.json must contain a 'model' section with a 'class' key.\n"
            "Example: {\"model\": {\"class\": \"president.models.single_meld_mlp.SingleMeldMLP\"}}"
        )
    cfg["model"] = model_cfg

    return cfg


def resolve_data_config(cfg: dict, multi_meld: bool) -> dict:
    """Fill in default data config keys depending on generation mode."""
    data_cfg = cfg.get("data", {})
    defaults = DEFAULT_EPISODE_DATA_CONFIG if multi_meld else DEFAULT_DATA_CONFIG
    for key, value in defaults.items():
        data_cfg.setdefault(key, value)
    return data_cfg


# ─────────────────────────────────────────────
# Player loading
# ─────────────────────────────────────────────

def _load_class(dotted: str):
    module_path, class_name = dotted.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ModuleNotFoundError, AttributeError) as exc:
        sys.exit(f"Cannot load class '{dotted}': {exc}")


def load_player(player_cfg: dict) -> AbstractPlayer:
    cls = _load_class(player_cfg["class"])
    return cls(player_cfg.get("name", "player"))


def load_model_class(model_cfg: dict):
    return _load_class(model_cfg["class"])


def load_opponents(cfg: dict) -> list[AbstractPlayer]:
    """
    Load three opponent players from config["opponents"].
    Falls back to three instances of the target player class if not specified.
    """
    opponents_cfg = cfg.get("opponents")
    if opponents_cfg:
        if len(opponents_cfg) != 3:
            sys.exit(f"config 'opponents' must have exactly 3 entries, got {len(opponents_cfg)}.")
        return [load_player(c) for c in opponents_cfg]
    # Default: three clones of the target player
    target_cfg = cfg["target_player"]
    return [
        load_player({**target_cfg, "name": f"opp_{i}"})
        for i in range(3)
    ]


def simple_player_setup(player: AbstractPlayer, hand: list, target) -> None:
    """Default state-setup for players that only need _hand and target_meld."""
    player._hand = sorted(hand, key=lambda c: c.get_index())
    player.target_meld = target


# ─────────────────────────────────────────────
# Hand generation (synthetic mode)
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

def encode_action(meld) -> np.ndarray:
    """55-class one-hot: 0-53 are melds, 54 is pass."""
    vec = np.zeros(ACTION_BITS, dtype=np.int8)
    if meld is None or not meld.cards:
        vec[54] = 1
    else:
        vec[:54] = StateEncoder.encode_meld(meld)
    return vec


def query_player(
    player: AbstractPlayer,
    hand: list,
    target,
    setup_fn: PlayerSetupFn = simple_player_setup,
) -> np.ndarray:
    setup_fn(player, hand, target)
    return encode_action(player.play())


# ─────────────────────────────────────────────
# Dataset builders — synthetic mode
# ─────────────────────────────────────────────

def build_training_set(
    n_hands: int,
    player: AbstractPlayer,
    model_cls,
    setup_fn: PlayerSetupFn = simple_player_setup,
) -> tuple:
    """
    Exhaustive: every valid (hand, target) pair for each hand.
    X shape is determined by model_cls.encode_state().
    """
    X, Y = [], []
    for _ in range(n_hands):
        hand = random_hand(random_hand_size())
        for target in valid_targets_for_hand(hand):
            state_enc  = model_cls.encode_state(hand, target)
            action_enc = query_player(player, hand, target, setup_fn)
            X.append(state_enc)
            Y.append(action_enc)
    return np.array(X, dtype=np.int8), np.array(Y, dtype=np.int8)


def build_random_set(
    n_hands: int,
    player: AbstractPlayer,
    model_cls,
    setup_fn: PlayerSetupFn = simple_player_setup,
) -> tuple:
    """
    Random: one random target per hand.
    Used for validation and test sets.
    X shape is determined by model_cls.encode_state().
    """
    X, Y = [], []
    for _ in range(n_hands):
        hand   = random_hand(random_hand_size())
        target = random_target_for_hand(hand)
        state_enc  = model_cls.encode_state(hand, target)
        action_enc = query_player(player, hand, target, setup_fn)
        X.append(state_enc)
        Y.append(action_enc)
    return np.array(X, dtype=np.int8), np.array(Y, dtype=np.int8)


# ─────────────────────────────────────────────
# Dataset builder — episode mode
# ─────────────────────────────────────────────

def build_episode_set(
    n_episodes: int,
    target_player: AbstractPlayer,
    opponents: list[AbstractPlayer],
    model_cls,
) -> tuple:
    """
    Run n_episodes real games and record every decision made by target_player.

    Opponents are the three other players at the table.
    Uses CloneDataCollector as a GameMaster listener to capture
    (hand, meld_context, action) at each of the target player's turns.

    Returns (X, Y, episode_records) where:
        X, Y            — encoded arrays for MLP training
        episode_records — list[EpisodeRecord] for future RNN training
    """
    from president.core.GameMaster import GameMaster, IllegalPlayPolicy
    from president.training.supervised.clone_data_collector import CloneDataCollector

    collector = CloneDataCollector(type(target_player))

    gm = GameMaster(policy=IllegalPlayPolicy.PENALISE)
    gm.add_listener(collector)   # before add_player so seats are registered
    gm.add_player(target_player)
    for opp in opponents:
        gm.add_player(opp)

    gm.start(number_of_rounds=n_episodes)
    done = False
    while not done:
        done = gm.step()

    X, Y = [], []
    for dp in collector.flat_decision_points():
        x = model_cls.encode_state(dp.hand, dp.melds)
        y = encode_action(dp.action)
        X.append(x)
        Y.append(y)

    return np.array(X, dtype=np.int8), np.array(Y, dtype=np.int8), collector.episode_records


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

    exp_dir = (EXPERIMENTS_DIR / args.experiment).resolve()
    if not exp_dir.is_dir():
        sys.exit(f"Experiment directory not found: {exp_dir}")

    data_dir = exp_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # ── Config ────────────────────────────────
    cfg        = load_config(exp_dir)
    player_cfg = cfg["target_player"]
    model_cfg  = cfg["model"]
    model_cls  = load_model_class(model_cfg)

    multi_meld = getattr(model_cls, "MULTI_MELD", False)
    data_cfg   = resolve_data_config(cfg, multi_meld)

    # ── Skip if data already complete ─────────
    required_files = DATA_FILES + (EPISODE_FILES if multi_meld else [])
    existing = [f for f in required_files if (data_dir / f).exists()]
    if len(existing) == len(required_files) and not args.force:
        print(f"Data already exists in {data_dir} — skipping generation.")
        print("Pass --force to regenerate.")
        return

    if args.force and existing:
        print(f"--force passed; regenerating {len(existing)} existing file(s).")

    print(f"Experiment    : {args.experiment}")
    print(f"Target player : {player_cfg['class']} (name={player_cfg['name']!r})")
    print(f"Model class   : {model_cfg['class']}")
    print(f"Mode          : {'episode-based' if multi_meld else 'synthetic'}")
    print(f"Data config   : {json.dumps(data_cfg, indent=2)}\n")

    random.seed(data_cfg["seed"])
    np.random.seed(data_cfg["seed"])

    # ── Generate ──────────────────────────────
    if multi_meld:
        train_ep = data_cfg["train_episodes"]
        val_ep   = data_cfg["val_episodes"]
        test_ep  = data_cfg["test_episodes"]

        print(f"Generating training set   ({train_ep:,} episodes)...")
        X_train, Y_train, ep_train = build_episode_set(
            train_ep, load_player(player_cfg), load_opponents(cfg), model_cls
        )

        print(f"Generating validation set ({val_ep:,} episodes)...")
        X_val, Y_val, ep_val = build_episode_set(
            val_ep, load_player(player_cfg), load_opponents(cfg), model_cls
        )

        print(f"Generating test set       ({test_ep:,} episodes)...")
        X_test, Y_test, ep_test = build_episode_set(
            test_ep, load_player(player_cfg), load_opponents(cfg), model_cls
        )

    else:
        setup_fn = simple_player_setup
        player   = load_player(player_cfg)

        print(f"Generating training set   ({data_cfg['train_hands']:,} hands, exhaustive)...")
        X_train, Y_train = build_training_set(data_cfg["train_hands"], player, model_cls, setup_fn)

        print(f"Generating validation set ({data_cfg['val_hands']:,} hands, random)...")
        X_val, Y_val = build_random_set(data_cfg["val_hands"], player, model_cls, setup_fn)

        print(f"Generating test set       ({data_cfg['test_hands']:,} hands, random)...")
        X_test, Y_test = build_random_set(data_cfg["test_hands"], player, model_cls, setup_fn)

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

    if multi_meld:
        for path, episodes in [
            (data_dir / "episodes_train.pkl", ep_train),
            (data_dir / "episodes_val.pkl",   ep_val),
            (data_dir / "episodes_test.pkl",  ep_test),
        ]:
            with open(path, "wb") as f:
                pickle.dump(episodes, f)

    print(f"\nSaved to {data_dir}")


if __name__ == "__main__":
    main()
