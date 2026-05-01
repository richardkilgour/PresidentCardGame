#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline game mode for the console UI.

Loads config, builds the GameMaster with the configured players, attaches an
EpisodeSave for trajectory recording, then runs the step loop until done.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

import sys

from president.core.EpisodeSave import EpisodeSave
from president.core.GameMaster import GameMaster
from president.core.GameSave import GameSave
from president.core.PlayerRegistry import PlayerRegistry
from president.players.HumanPlayer import QuitGame
from president.players.PlayerConsole import PlayerConsole

_CONFIG_PATH = Path(__file__).parents[2] / 'config' / 'config.yaml'
_LOG_PATH    = Path(__file__).parents[2] / 'logs' / 'test.log'


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
        print(GameSave.display(args.restore))
        record = GameSave.restore_combined(args.restore, gm, registry)
        gm.add_listener(record)
        gm.set_record(record)
        print("Restore complete — resuming game.")
    else:
        record = EpisodeSave(gm)
        for key in ['player1', 'player2', 'player3', 'player4']:
            p = config[key]
            if p.get('console', False):
                gm.add_player(PlayerConsole(p['name']))
            else:
                gm.make_player(p['type'], p['name'])
        gm.start(number_of_rounds=1000)
        gm.add_listener(record)
        gm.set_record(record)

    done = False
    while not done:
        try:
            done = gm.step()
        except QuitGame:
            path = GameSave.stamped_path("quit_save")
            GameSave(gm).save_combined(record, path)
            print(f'\n  Game saved to {path}. Resume with: --restore {path}')
            sys.exit(0)

    print("Game complete.")
    print(gm.position_stats_str())
