#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Play a game of Asshole locally
Creates a GM and some players, and lets the GM control the game
"""
import logging
import yaml
import sys


from asshole.GameMaster import GameMaster
from asshole.player.ConsolePlayer import ConsolePlayer
from asshole.player.PlayerSimple import PlayerSimple
from asshole.player.PlayerHolder import PlayerHolder
from asshole.player.PlayerSplitter import PlayerSplitter


def main():
    # logging.basicConfig(handlers=[logging.FileHandler('test.log', 'w', 'utf-8')], level=logging.DEBUG)
    logging.basicConfig(handlers=[logging.FileHandler('test.log', 'w', 'utf-8')], level=logging.NOTSET)

    gm = GameMaster()

    config = yaml.safe_load(open("./config.yaml"))
    players = [config['player1'], config['player2'], config['player3'], config['player4'], ]
    for i, p in enumerate(players):
        player_class = getattr(sys.modules[__name__], p['type'])
        gm.make_player(player_class, p['name'])
    gm.play(number_of_rounds=100)


if __name__ == '__main__':
    main()
