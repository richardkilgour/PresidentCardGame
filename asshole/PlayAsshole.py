#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Play a game of Asshole locally
Creates a GM and some players, and lets the GM control the game
"""
import logging

from asshole.GameMaster import GameMaster
from asshole.player.HumanPlayer import HumanPlayer
from asshole.player.PlayerSimple import PlayerSimple
from asshole.player.PlayerHolder import PlayerHolder
from asshole.player.PlayerSplitter import PlayerSplitter
# from asshole.player.TensorflowPlayer import TensorflowPlayer


def main():
    # logging.basicConfig(handlers=[logging.FileHandler('test.log', 'w', 'utf-8')], level=logging.DEBUG)
    logging.basicConfig(handlers=[logging.FileHandler('test.log', 'w', 'utf-8')], level=logging.NOTSET)
    gm = GameMaster()
    # Players can be player_simple, player_splitter, player_holder, human_player
    gm.make_player(PlayerHolder, "Richard")
    # gm.make_player(TensorflowPlayer, "Silvia")
    gm.make_player(PlayerSplitter, "Silvia")
    #gm.make_player(HumanPlayer, "Silvia")
    gm.make_player(PlayerSplitter, "Sara")
    gm.make_player(PlayerSimple, "snAkbar")
    gm.play(number_of_rounds=1000)


if __name__ == '__main__':
    main()
