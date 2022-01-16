import logging
from random import shuffle
import numpy as np
import pickle

from asshole.cards.PlayingCard import PlayingCard
# Needed for deserialization eval function
from asshole.Episode import Episode


class GameMaster:
    """A class to control all the action"""

    def __init__(self, deck_size=54):
        self.players = []
        self.positions = []
        # This is a list of all the card objects - they will me moved around
        self.deck = [PlayingCard(i) for i in range(deck_size)]
        self.listener_list = []

    def clear(self):
        self.players = []
        self.positions = []
        self.listener_list = []

    # State is a list of INT16. Each player is 7 ints, which are a bit-mask for:
    # The Hand (54 bits), the state (2 bits), the position (4 bits) and the meld (54 bits)
    def save_state(self):
        """"Encode each hand, each (current) meld, state an positions for each player"""
        # List of np arrays
        game_state = []
        player_names = []
        player_types = []
        for player in self.players:
            game_state.append(player.encode())
            player_names.append(player.name)
            player_types.append(player.__class__.__name__)

        # Who is active? Always player 0!!!
        # active_index = gm.player.index(gm.active_players[0])
        game_state[0][2, 27] = 1
        serialized = pickle.dumps((player_names, player_types, game_state), protocol=0)  # protocol 0 is printable ASCII
        return serialized

    def restore_state(self, serialized):
        deserialized_a = pickle.loads(serialized)
        # In any case, restore the gm to a blank state
        self.clear()
        # TODO: Dear god fix this.
        # first comes names, then class names, then the hand nad meld
        player_names = deserialized_a[0]
        player_types = deserialized_a[1]
        game_state = deserialized_a[2]
        self.next_to_play = []
        for i, name in enumerate(player_names):
            # TODO: Nasty - turn the name into a class
            player_class = eval(player_types[i])
            player = self.make_player(player_class, name)
            # print("shape of the player {}".format(game_state[i].shape))
            hand_meld = np.split(game_state[1], 2, axis=1)
            hand = hand_meld[0]
            meld = hand_meld[0]
            player._hand = player.decode_hand(hand)
            if game_state[i][2, 13] == 1:
                player.set_status("passed")
            elif game_state[i][3, 13] == 1:
                player.set_status("waiting")
            else:
                player.set_status(player.decode_hand(meld))
        return self

    def snapshot(self):
        # TODO: Something is broken here - all the cards and the current meld is messed up
        return
        state = self.save_state()
        # print("Serialized as {}".format(state))
        self.restore_state(state)

    def add_listener(self, listener):
        self.listener_list.append(listener)

    def notify_listeners(self, notify_func_name, *args):
        for p in self.listener_list:
            getattr(p, notify_func_name)(*args)

    def get_player_status(self, index):
        """"
        What is the status of the player of the given index?
        Waiting (not yet played)
        Passed
        Finished (no cards left)
        Played (just return the cards played)
        """
        return self.players[index].get_status()

    def make_player(self, player_type, name=None):
        # print("Making a {} player names {}".format(player_type,name))
        # player_type is a class or a string (for a saved / serialised player)
        if not name:
            name = "Player " + len(self.players)

        if len(self.players) < 4:
            new_player = player_type(name)
            # Notify everyone of the new player (This includes the player itself, since it's now  registered as a listener)
            self.notify_listeners("notify_player_joined", new_player)
            # Notify the new player of the other players
            for p in self.players:
                new_player.notify_player_joined(p)
            self.players.append(new_player)
            # TODO: What is a player was passed? it has a name...
        else:
            raise Exception("Too many Players")
        # return the new guy
        self.add_listener(self.players[-1])
        return self.players[-1]

    def play(self, number_of_rounds=100, preset_hands=None):
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
            # Do a round
            this_episode = Episode(self.players, self.positions, self.deck, self.listener_list)
            self.positions = this_episode.play_episode()
            assert (len(self.deck) == 54)
            # Keep some stats
            round_count += 1
            if round_count > number_of_rounds:
                self.report_position_stats()
                self.remove_worst_player()

    def report_position_stats(self):
        """ Print the total times in each position"""
        for p in self.players:
            print("{} was King {}; Prince {}; Citizen {} and Asshole {}. Score = {}".format(p.name,
                                                                                            p.position_count[0],
                                                                                            p.position_count[1],
                                                                                            p.position_count[2],
                                                                                            p.position_count[3],
                                                                                            p.get_score()))

    def remove_worst_player(self):
        """Worst player quits the game!!!"""
        worst_player = None
        lowest_score = 0
        for p in self.players:
            score = p.get_score()
            if score < lowest_score:
                worst_player = p
                lowest_score = score
        print("{} is pissed off and quits".format(worst_player.name))
        self.players.remove(p)
