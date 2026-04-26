#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline game mode for the console UI.

Loads config, builds the GameMaster with the configured players, attaches a
GameRecord for save/restore, then runs the step loop until the game is done.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

from president.core.GameMaster import GameMaster
from president.core.GameRecord import GameRecord
from president.core.PlayerRegistry import PlayerRegistry
from president.players.ConsolePlayer import ConsolePlayer

_CONFIG_PATH = Path(__file__).parents[3] / 'config' / 'config.yaml'
_LOG_PATH    = Path(__file__).parents[3] / 'test.log'


def run_offline(args) -> None:
    logging.basicConfig(
        handlers=[logging.FileHandler(_LOG_PATH, 'w', 'utf-8')],
        level=logging.NOTSET,
    )

    config   = yaml.safe_load(_CONFIG_PATH.read_text())
    registry = PlayerRegistry.from_config(config)
    gm       = GameMaster(registry=registry)

    if args.restore:
        print(f"Restoring from {args.restore}...")
        print(GameRecord.display(args.restore))
        record = GameRecord.restore(args.restore, gm, registry)
        print("Restore complete — resuming game.")
    else:
        record = GameRecord(gm)
        for key in ['player1', 'player2', 'player3', 'player4']:
            p = config[key]
            if p.get('console', False):
                gm.add_player(ConsolePlayer(p['name']))
            else:
                gm.make_player(p['type'], p['name'])
        gm.start(number_of_rounds=1000)

    gm.add_listener(record)
    gm.set_record(record)

    for player in gm.player_manager.players:
        if isinstance(player, ConsolePlayer):
            player.set_record(record)

    done = False
    while not done:
        done = gm.step()

    record.mark_complete()
    print("Game complete.")
    print(gm.position_stats_str())
