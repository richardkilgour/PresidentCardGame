import logging
from random import shuffle

from asshole.core.AbstractPlayer import AbstractPlayer
from asshole.core.PlayingCard import PlayingCard
# Needed for deserialization eval function
from asshole.core.Episode import Episode, State


class GameMaster:
    """A class to control all the action"""

    def __init__(self, deck_size=54):
        self.players = []
        self.positions = []
        # This is a list of all the card objects - they will be moved around
        self.deck_size = deck_size
        self.deck = [PlayingCard(i) for i in range(deck_size)]
        self.listener_list = []
        self.number_of_rounds = None
        self.round_number = 0

    def clear(self):
        self.players = []
        self.positions = []
        self.listener_list = []

    def add_listener(self, listener):
        self.listener_list.append(listener)

    def notify_listeners(self, notify_func_name, *args):
        for p in self.listener_list:
            getattr(p, notify_func_name)(*args)

    def get_player_status(self, index):
        """
        What is the status of the players of the given index?
        Waiting (not yet played)
        Passed
        Finished (no cards left)
        Played (just return the cards played)
        """
        return self.players[index].get_status()

    def add_player(self, player: AbstractPlayer, name=None):
        if len(self.players) < 4:
            # Notify everyone of the new players
            # This includes the players itself, since it's now registered as a listener
            self.notify_listeners("notify_player_joined", player)
            # Notify the new players of the other players
            for p in self.players:
                player.notify_player_joined(p)
            self.players.append(player)
            self.add_listener(player)
        else:
            raise Exception("Too many Players")

    def make_player(self, player_type, name=None):
        # print("Making a {} players names {}".format(player_type,name))
        # player_type is a class or a string (for a saved / serialised players)
        if not name:
            name = f'Player {len(self.players)}'
        new_player = player_type(name)
        self.add_player(new_player, name)
        # return the new guy
        return self.players[-1]

    def reset(self, preset_hands=None):
        logging.info(f"--- Start of a new Game ---{self.positions=}")
        # A client can shuffle the cards themselves (for testing... or cheating?)
        # Shuffle the cards
        if not preset_hands:
            shuffle(self.deck)
        self.episode = Episode(self.players, self.positions, self.deck, self.listener_list)


    def start(self, number_of_rounds=100, preset_hands=None):
        """
        Play a bunch of hands, after which a player leaves and the game stops
        Some basics stats on the game are then printed
        number_of_rounds = None will never stop
        preset_hands is a prepared deck (for loading a game, testing purposes or tournament play)
        """
        if len(self.players) != 4:
            raise Exception("Not enough Players")
        self.round_number = 0
        self.number_of_rounds = number_of_rounds
        # The initial hand has no positions
        self.reset(preset_hands=preset_hands)

    def step(self) -> bool:
        # Return True if finished all rounds, else False
        if self.episode.state == State.FINISHED:
            # Set up a new round
            self.positions = self.episode.positions
            logging.info(f"--- Round Finished with positions {self.positions} ---")
            self.reset()
            assert (len(self.deck) == 54)
            # Keep some stats?
            self.round_number += 1
            if self.number_of_rounds and self.round_number > self.number_of_rounds:
                print(self.position_stats_str())
                self.remove_worst_player()
                return True
        else:
            # May be blocking if a player does not play
            self.episode.step()
        return False

    def position_stats_str(self):
        """ Return the total times in each position as a string"""
        retval = ""
        for p in self.players:
            retval += f'{p.name} was King {p.position_count[0]}; Prince {p.position_count[1]}; ' \
                      f'Citizen {p.position_count[2]} and Asshole {p.position_count[3]}. Score = {p.get_score()}\n'
        return retval

    def remove_worst_player(self):
        """Worst players quits the game!!!"""
        worst_player = None
        lowest_score = 0
        for p in self.players:
            score = p.get_score()
            if score < lowest_score:
                worst_player = p
                lowest_score = score
        print(f'{worst_player.name} is pissed off and quits')
        self.players.remove(worst_player)
