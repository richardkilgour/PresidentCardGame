import logging
from random import shuffle

from asshole.core.PlayingCard import PlayingCard
# Needed for deserialization eval function
from asshole.gym_env.Episode import Episode, State


class GameMaster:
    """A class to control all the action"""

    def __init__(self, deck_size=54):
        self.players = []
        self.positions = []
        # This is a list of all the card objects - they will me moved around
        self.deck_size = deck_size
        self.deck = [PlayingCard(i) for i in range(deck_size)]
        self.listener_list = []

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

    def make_player(self, player_type, name=None):
        # print("Making a {} players names {}".format(player_type,name))
        # player_type is a class or a string (for a saved / serialised players)
        if not name:
            name = f'Player {len(self.players)}'

        if len(self.players) < 4:
            new_player = player_type(name)
            # Notify everyone of the new players
            # This includes the players itself, since it's now registered as a listener
            self.notify_listeners("notify_player_joined", new_player)
            # Notify the new players of the other players
            for p in self.players:
                new_player.notify_player_joined(p)
            self.players.append(new_player)
        else:
            raise Exception("Too many Players")
        # return the new guy
        self.add_listener(self.players[-1])
        return self.players[-1]

    def play(self, number_of_rounds=100, preset_hands=None):
        """
        Play a bunch of hands, after which a players leaves and the game stops
        Some basics stats on the game are then printed
        number_of_rounds = None will never stop
        preset_hands is a prepared deck (for testing purposes or tourniment play)
        """
        if len(self.players) != 4:
            raise Exception("Not enough Players")
        round_count = 0

        # Play an initial hand, with no positions
        print("NEW GAME - no positions")

        # loop until someone leaves the game
        while len(self.players) == 4:
            logging.info("--- Start of a new Hand ---")
            # A client can shuffle the cards themselves (for testing... or cheating?)
            # Shuffle the cards
            if not preset_hands:
                shuffle(self.deck)
            # Do an episode
            this_episode = Episode(self.players, self.positions, self.deck, self.listener_list)
            while this_episode.state != State.FINISHED:
                this_episode.play()
            self.positions = this_episode.positions
            assert (len(self.deck) == 54)
            # Keep some stats
            round_count += 1
            if number_of_rounds and round_count > number_of_rounds:
                print(self.position_stats_str())
                self.remove_worst_player()

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
