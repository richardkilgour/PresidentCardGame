#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Play a game of PresidentCardGame locally.
Creates a GM and some players, and lets the GM control the game.

Usage:
    python PlayLocally.py                        # start a new game
    python PlayLocally.py --restore crash.json   # resume from a checkpoint
    python PlayLocally.py --display crash.json   # display a checkpoint without running
"""
import argparse
import importlib
import logging
import yaml

from president.core.GameMaster import GameMaster
from president.core.GameCheckpoint import GameCheckpoint


def load_player_class(player_config: dict):
    module = importlib.import_module('players.' + player_config['type'])
    return getattr(module, player_config['type'])


def build_player_registry(config: dict) -> dict:
    """Load all player classes from config and return a registry mapping type name to class."""
    registry = {}
    for key in ['player1', 'player2', 'player3', 'player4']:
        p = config[key]
        registry[p['type']] = load_player_class(p)
    return registry


def main():
    parser = argparse.ArgumentParser(description="Play PresidentCardGame locally.")
    parser.add_argument(
        '--restore',
        type=str,
        metavar='FILE',
        help='Resume a game from a checkpoint file.'
    )
    parser.add_argument(
        '--display',
        type=str,
        metavar='FILE',
        help='Display a checkpoint file in human-readable form and exit.'
    )
    args = parser.parse_args()

    # --display needs no game setup at all
    if args.display:
        print(GameCheckpoint.display(args.display))
        return

    logging.basicConfig(
        handlers=[logging.FileHandler('test.log', 'w', 'utf-8')],
        level=logging.NOTSET
    )

    config = yaml.safe_load(open("config/config.yaml"))
    player_registry = build_player_registry(config)

    gm = GameMaster()
    checkpoint = GameCheckpoint(gm)

    if args.restore:
        print(f"Restoring from {args.restore}...")
        print(GameCheckpoint.display(args.restore))  # Show state before restoring
        GameCheckpoint.restore(args.restore, gm, player_registry)
        print("Restore complete — resuming game.")
    else:
        for p in [config['player1'], config['player2'], config['player3'], config['player4']]:
            gm.make_player(player_registry[p['type']], p['name'])
        gm.start(number_of_rounds=1000)

    done = False
    while not done:
        try:
            done = gm.step()
        except Exception:
            checkpoint.save_on_error(checkpoint.stamped_path("crash"))
            logging.error("Game crashed — checkpoint saved to crash.json")
            break

    if done:
        print("Game complete.")
        print(gm.position_stats_str())


if __name__ == '__main__':
    main()