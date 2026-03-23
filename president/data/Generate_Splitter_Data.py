#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate training, validation, and test datasets for cloning PlayerSplitter.

Training:   exhaustive (hand, target) pairs for N hands
Validation: random     (hand, target) pairs for M hands
Test:       random     (hand, target) pairs for M hands

Hands in all three sets are disjoint.
"""
import random
from pathlib import Path
import numpy as np
from collections import Counter
from president.players.PlayerSplitter import PlayerSplitter
from president.core.PlayingCard import PlayingCard
from president.core.Meld import Meld
from president.core.StateEncoder import StateEncoder

ACTION_BITS = 55  # 0-53 melds, 54 = pass
MELD_BITS   = 54
JOKER_VALUE = 13

DATA = Path(__file__).parent.parent / "data"


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
# Oracle
# ─────────────────────────────────────────────

def query_oracle(player: PlayerSplitter, hand: list, target) -> np.ndarray:
    """Ask PlayerSplitter what it would play given hand and target."""
    player._hand = list(hand)
    player._hand.sort(key=lambda c: c.get_index())
    player.target_meld = target
    action = player.play()
    return encode_action(action)


# ─────────────────────────────────────────────
# Dataset builders
# ─────────────────────────────────────────────

def build_training_set(n_hands: int, player: PlayerSplitter) -> tuple:
    """
    Exhaustive: every valid (hand, target) pair for each hand.
    Returns X (N, 108), Y (N, 55).
    """
    X, Y = [], []
    for _ in range(n_hands):
        hand = random_hand(random_hand_size())
        for target in valid_targets_for_hand(hand):
            hand_enc, target_enc = encode_example(hand, target)
            action_enc = query_oracle(player, hand, target)
            X.append(np.concatenate([hand_enc, target_enc]))
            Y.append(action_enc)
    return np.array(X, dtype=np.int8), np.array(Y, dtype=np.int8)


def build_random_set(n_hands: int, player: PlayerSplitter) -> tuple:
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
        action_enc = query_oracle(player, hand, target)
        X.append(np.concatenate([hand_enc, target_enc]))
        Y.append(action_enc)
    return np.array(X, dtype=np.int8), np.array(Y, dtype=np.int8)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    player = PlayerSplitter("oracle")

    print("Generating training set  (10 000 hands, exhaustive)...")
    X_train, Y_train = build_training_set(n_hands=10_000, player=player)

    print("Generating validation set (12 000 hands, random)...")
    X_val, Y_val = build_random_set(n_hands=12_000, player=player)

    print("Generating test set       (12 000 hands, random)...")
    X_test, Y_test = build_random_set(n_hands=12_000, player=player)

    print(f"\nTraining:   {len(X_train):>7,} examples")
    print(f"Validation: {len(X_val):>7,} examples")
    print(f"Test:       {len(X_test):>7,} examples")

    np.save(DATA / "X_train.npy", X_train)
    np.save(DATA / "Y_train.npy", Y_train)
    np.save(DATA / "X_val.npy",   X_val)
    np.save(DATA / "Y_val.npy",   Y_val)
    np.save(DATA / "X_test.npy",  X_test)
    np.save(DATA / "Y_test.npy",  Y_test)

    print(f"\nSaved to {DATA}")
