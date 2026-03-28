#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tournament runner for President.

Reads the same config/config.yaml as PlayLocally.

Each tournament consists of 100 episodes:
  - 1 neutral position combination × 4 hand rotations
  - 24 permutation position combinations × 4 hand rotations

For each position combination a fresh set of hands is dealt, then
rotated across all 4 seats so every player plays every hand once.
This ensures identical policies converge to equal scores.

Usage:
    python president/models/Tournament.py
    python president/models/Tournament.py --config config/my_config.yaml
    python president/models/Tournament.py --tournaments 50
"""
from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from president.core.DeckManager import DeckManager
from president.core.GameMaster import GameMaster, IllegalPlayPolicy
from president.core.PlayerRegistry import PlayerRegistry
from president.core.TournamentDeck import TournamentDeck

RANK_NAMES   = ["President", "Vice-President", "Citizen", "Scumbag"]
RANK_REWARDS = {0: 2, 1: 1, 2: -1, 3: -2}


# ─────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────

@dataclass
class PlayerResult:
    name:           str
    policy:         str
    position_count: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    by_start:       list[list[int]] = field(
        default_factory=lambda: [[0] * 4 for _ in range(4)]
    )

    def record(self, starting_rank: int | None, final_rank: int):
        self.position_count[final_rank] += 1
        if starting_rank is not None:
            self.by_start[starting_rank][final_rank] += 1

    @property
    def score(self) -> int:
        return sum(RANK_REWARDS[r] * c
                   for r, c in enumerate(self.position_count))

    def summary(self) -> str:
        lines = [
            f"{self.name} ({self.policy})  "
            f"score={self.score}  "
            f"counts={self.position_count}"
        ]
        for sr in range(4):
            counts = self.by_start[sr]
            if any(counts):
                lines.append(
                    f"  started as {RANK_NAMES[sr]:16s} → "
                    f"P={counts[0]:3d}  VP={counts[1]:3d}  "
                    f"C={counts[2]:3d}  S={counts[3]:3d}"
                )
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Tournament
# ─────────────────────────────────────────────

class Tournament:
    """
    Runs a series of tournaments using players defined in config.yaml.
    Each tournament = 25 position combos × 4 hand rotations = 100 episodes.
    """

    def __init__(self, config: dict, n_tournaments: int = 40):
        self.config = config
        self.n_tournaments = n_tournaments
        self.registry = PlayerRegistry.from_config(config)
        self._permutations = list(itertools.permutations(range(4)))
        # Set seed for debugging
        self._deck = TournamentDeck(seed=42)

    def run(self) -> list[PlayerResult]:
        players = []
        for key in ['player1', 'player2', 'player3', 'player4']:
            p = self.config[key]
            players.append(self.registry.create(p['type'], p['name']))

        results = [
            PlayerResult(name=p.name, policy=p.__class__.__name__)
            for p in players
        ]

        for t in range(self.n_tournaments):
            print(f"Tournament {t + 1}/{self.n_tournaments}...")
            self._run_tournament(players, results)

        return results

    def _run_tournament(self, players: list,
                        results: list[PlayerResult]) -> None:
        """
        Run one tournament: 25 position combos × 4 hand rotations = 100 episodes.
        """
        # Use the same deck and hands for the whole tournament.
        # Just rotate the hands for each combo
        self._deck.new_deal()

        # ── Neutral position combo ────────────────────────────────────
        self._run_combo(
            players=players,
            results=results,
            starting_ranks=[],
            starting_rank_indices=None,
        )

        # Establish base ranks from a fresh neutral episode for permutations
        # Use the first neutral rotation's result as the base
        base_ranks = self._last_base_ranks

        # ── 24 permutation combos ─────────────────────────────────────
        for perm in self._permutations:
            starting_ranks = [base_ranks[perm[i]] for i in range(4)]
            self._run_combo(
                players=players,
                results=results,
                starting_ranks=starting_ranks,
                starting_rank_indices=list(perm),
            )

    def _run_combo(self, players, results, starting_ranks,
                   starting_rank_indices):
        """Run 4 rotated episodes for one position combination."""
        for rotation in range(4):
            final_ranks = self._run_episode(
                players=players,
                starting_ranks=starting_ranks,
            )

            if starting_rank_indices is None and rotation == 0:
                self._last_base_ranks = self._ranks_to_player_list(
                    players, final_ranks
                )

            for i in range(4):
                sr = starting_rank_indices[i] \
                    if starting_rank_indices is not None else None
                results[i].record(starting_rank=sr, final_rank=final_ranks[i])
            self._deck.rotate()


    def _run_episode(self, players, starting_ranks):
        for p in players:
            p._hand = []
            p.target_meld = None

        gm = GameMaster(
            registry=self.registry,
            policy=IllegalPlayPolicy.PENALISE,
        )
        # Inject the TournamentDeck so deal_cards uses preset hands
        gm.deck = self._deck

        for p in players:
            gm.add_player(p)

        gm.start(number_of_rounds=1, positions=starting_ranks)

        done = False
        while not done:
            done = gm.step()

        final_rank_list = gm.episode.ranks
        return [final_rank_list.index(p) for p in players]

    def _deal_hands(self, deck: DeckManager) -> list[list]:
        """
        Deal cards from the deck into 4 hands without using the full
        game machinery — returns raw card lists for preset use.
        """
        from president.core.PlayingCard import PlayingCard
        cards = [PlayingCard(i) for i in range(54)]
        # Shuffle order mirrors DeckManager internals
        import random
        random.shuffle(cards)
        # Deal 13-14 cards each (54 cards, 4 players)
        hands = [[], [], [], []]
        for i, card in enumerate(cards):
            hands[i % 4].append(card)
        for hand in hands:
            hand.sort(key=lambda c: c.get_index())
        return hands


    @staticmethod
    def _ranks_to_player_list(players: list,
                               final_ranks: list[int]) -> list:
        """
        Convert per-player rank indices to ordered
        [player_at_rank_0, ..., player_at_rank_3].
        """
        ordered = [None] * 4
        for i, rank in enumerate(final_ranks):
            ordered[rank] = players[i]
        return ordered


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a President tournament."
    )

    default_config_path = "config/config.yaml"
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path,
        help=f"Path to config file (default: {default_config_path})",
    )

    default_num_tournaments = 10
    parser.add_argument(
        "--tournaments",
        type=int,
        default=default_num_tournaments,
        help=f"Number of tournaments to run (default: {default_num_tournaments})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        handlers=[logging.FileHandler("tournament.log", "w", "utf-8")],
        level=logging.WARNING,
    )

    config = yaml.safe_load(open(args.config))

    t = Tournament(config=config, n_tournaments=args.tournaments)
    results = t.run()

    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    for r in sorted(results, key=lambda x: -x.score):
        print()
        print(r.summary())
