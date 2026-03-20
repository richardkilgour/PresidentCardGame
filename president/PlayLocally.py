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
import logging

import yaml

from president.core.GameCheckpoint import GameCheckpoint
from president.core.GameMaster import GameMaster
from president.core.PlayerRegistry import PlayerRegistry
from president.players.ConsolePlayer import ConsolePlayer

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
    registry = PlayerRegistry.from_config(config)
    gm = GameMaster(registry=registry)
    checkpoint = GameCheckpoint(gm)
    gm.set_checkpoint(checkpoint)

    if args.restore:
        print(f"Restoring from {args.restore}...")
        print(GameCheckpoint.display(args.restore))
        GameCheckpoint.restore(args.restore, gm, registry)
        print("Restore complete — resuming game.")
    else:
        for key in ['player1', 'player2', 'player3', 'player4']:
            p = config[key]
            if p.get('console', False):
                player = ConsolePlayer(p['name'])
                player.set_checkpoint(checkpoint)
                gm.add_player(player)
            else:
                gm.make_player(p['type'], p['name'])
        gm.start(number_of_rounds=1000)

    # Wire up checkpoint for console players after restore too
    for player in gm.player_manager.players:
        if isinstance(player, ConsolePlayer):
            player.set_checkpoint(checkpoint)

    done = False
    while not done:
        done = gm.step()

    if done:
        print("Game complete.")
        print(gm.position_stats_str())


if __name__ == '__main__':
    main()
