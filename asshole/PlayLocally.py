#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Play a game of Asshole locally
Creates a GM and some players, and lets the GM control the game
"""
import importlib
import logging
import yaml
from asshole.core.GameMaster import GameMaster


def main():
    logging.basicConfig(handlers=[logging.FileHandler('test.log', 'w', 'utf-8')], level=logging.NOTSET)

    gm = GameMaster()

    config = yaml.safe_load(open("utils/config.yaml"))
    players = [config['player1'], config['player2'], config['player3'], config['player4'], ]
    for i, p in enumerate(players):
        module_name = 'players.' + p['type']  # Assuming the module name matches the class name in lowercase
        # Dynamically import the module
        module = importlib.import_module(module_name)
        player_class = getattr(module, p['type'])
        gm.make_player(player_class, p['name'])
    gm.start(number_of_rounds=100)
    while not gm.step():
        pass



if __name__ == '__main__':
    main()
