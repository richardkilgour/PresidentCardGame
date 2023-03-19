#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class to control a single 'Episode':
Needs the players (and their starting positions), a deck, and listeners

Does this:
    Shuffles and deals (on behalf of a player)
    Enforces card swapping
    Play 'rounds' until everyone is finished
        Pick a starter
    Return the new positions

Remembers the current state (including the memory?)
That makes it a markov thingie

Needs to be told:
    What was just played

Should report on:
    Positions at the start of the round (if anyone cares)
    Who's turn is it?
    Highest current card played (and who played it)
    Who's still 'in' (or if they have passed)
        What everyone's last card was
    Who's retired (New king...)

Needs to know:
    How many cards are left in each hand (can query the players)
"""
import logging
from enum import Enum
from random import shuffle
import numpy as np
import pickle


class State(Enum):
    INITIALISED = 1
    DEALING = 2
    SWAPPING = 3
    ROUND_STARTING = 4
    PLAYING = 5
    HAND_WON = 6
    FINISHED = 7


class Episode:
    def __init__(self, players, positions, deck, listener_list):
        # All the played cards (for checking)
        self.discards = []
        # List of players in their positions from last round
        # Index 0 is King, Index 3 is Asshole
        # None if it's a new game and no positions have been established
        # The Episode will (re) populate this and return after the episode has run
        self.positions = positions
        self.deck = deck
        self.players = players
        # The highest current play (Meld) for each player
        # None is pass, and '␆' is not played yet
        self.current_melds = ['␆', '␆', '␆', '␆']
        # All the listeners
        self.listener_list = listener_list
        self.state = State.INITIALISED
        self.active_players = []

    def target_meld(self):
        """Find the highest set of cards currently played"""
        highest_meld = None
        for m in self.current_melds:
            if m and m != '␆' and (not highest_meld or m > highest_meld):
                highest_meld = m
        return highest_meld

    def swap_cards(self):
        """Swap cards if necessary"""
        if self.positions:
            # Asshole swaps 2 cards with King
            print(f'{self.positions[3].name} must give {self.positions[0].name} 2 cards')
            tribute = [self.positions[3]._hand[-2], self.positions[3]._hand[-1]]
            self.positions[3].surrender_cards(tribute, self.positions[0])
            discard = [self.positions[0]._hand[0], self.positions[0]._hand[1]]
            self.positions[0].surrender_cards(discard, self.positions[3])
            logging.debug(f'{self.positions[3].name} swapped {tribute[0]} and {tribute[1]} for '
                          f"{self.positions[0].name}'s cards {discard[0]} and {discard[1]}")
            # Prince swaps 1 card with Citizen
            print(f'{self.positions[2].name} must give {self.positions[1].name} 1 card')
            tribute = [self.positions[2]._hand[-1]]
            self.positions[2].surrender_cards(tribute, self.positions[1])
            discard = [self.positions[1]._hand[0]]
            self.positions[1].surrender_cards(discard, self.positions[2])
            logging.debug(
                f'{self.positions[2].name} swapped {tribute[0]} for {self.positions[1].name} cards {discard[0]}')
            for player in self.players:
                # Log everyone's hands after teh swap
                logging.debug(f'{player.name} has {player}')

    def pick_round_starter(self):
        """Decide who is the first player for the first round"""
        if self.positions:
            # Asshole goes first
            self.move_to_front(self.positions[3])
        else:
            # Probably first hand, so start with 3♠
            player_with_3_spades = self.find_card_holder(0, 0)
            self.move_to_front(player_with_3_spades)
        # Tell everyone who is starting the round
        self.notify_listeners("notify_hand_start", self.players[0])

    def notify_listeners(self, notify_func_name, *args):
        for p in self.listener_list:
            getattr(p, notify_func_name)(*args)

    def move_to_front(self, front_player):
        while self.players[0] != front_player:
            self.players.append(self.players.pop(0))

    def deal(self):
        # Check all the cards are home
        for player in self.players:
            assert (player.report_remaining_cards() == 0)
        assert (len(self.discards) == 0)

        # Deal the cards, starting at player 1 (assumes that player 0 is dealing)
        i = 1
        # Probably a more efficient way to do this, but...
        # Instead of dealing 0 to n, reverse and pop the last every time
        self.deck.reverse()
        while len(self.deck) > 0:
            self.players[i].card_to_hand(self.deck.pop())
            i = (i + 1) % 4

    def find_card_holder(self, target_card_value, target_card_suit):
        # Maybe it's not the first card in the hand, 'cause suits are not sorted
        for player in self.players:
            # Cards are sorted by value, so could do a binary search, but whatever
            for card in player._hand:
                if card.get_value() > target_card_value:
                    break
                if card.get_value() == target_card_value and + card.get_suit() == target_card_suit:
                    return player

    def players_with_cards(self):
        active_players = []

        for player in self.players:
            # Everyone is active unless they have played out already
            if player.report_remaining_cards() == 0:
                # 0 cards means they have a position
                assert (player in self.positions)
            else:
                active_players.append(player)
        return active_players

    def post_episode_checks(self):
        asshole = self.positions[-1]
        logging.info(f'{asshole.name} is the Asshole!!! Left with {asshole._hand}')
        # Flush the Asshole's hand
        while len(asshole._hand) > 0:
            self.discards.append(asshole._hand.pop())
        assert (len(self.discards) == 54)
        assert (len(self.deck) == 0)

        # Shift the discards back to deck
        while len(self.discards) > 0:
            self.deck.append(self.discards.pop())

        assert (len(self.positions) == 4)
        for i, p in enumerate(self.positions):
            print(f"{p.name} is ranked as {p.ranking_names[i]}")
        for p in self.players:
            assert (p.report_remaining_cards() == 0)

        assert (len(self.deck) == 54)

    def set_player_finished(self, player):
        """
        Someone just played out, so assign them a position,
        Technically they are still 'in' and their meld counts
        If the hands makes it back round, they will pass (and gloat)
        """
        logging.info(f"{player.name} played out and is ranked {player.ranking_names[len(self.positions)]}")
        player.set_position(len(self.positions))
        self.notify_listeners("notify_played_out", player, len(self.positions))
        self.positions.append(player)
        player_names = ''
        for x in self.positions:
            player_names = player_names + ' ' + x.name
        logging.info(f"Positions: {player_names}")

    def player_turn(self):
        """
        Let a player have a go
        Even the Asshole "wins" the last round, and their cards are drained
        """
        # Everyone who has cards is now active
        # This is a circular list:
        #   Players who pass
        #   Players that play are moved to the back
        # Round is over when one player remains active
        assert (len(self.active_players) > 0)

        # TODO: not pythonic
        player_names = ''
        for x in self.active_players:
            player_names = player_names + ' ' + x.name
        logging.info(f"Players who have not passed = {player_names}")
        # ================
        # Round Starts with some poor, lonely sod.
        # Send the hand finished code!!!
        # ================
        if len(self.active_players) == 1:
            self.set_player_finished(self.active_players[0])
            return

        # We have more than one active player, so let the current plyer play (or fall-though)
        player = self.active_players[0]
        if player.report_remaining_cards() == 0:
            # This payer has played out, but others have played on their last card
            self.active_players.remove(player)
            return

        logging.info(f'Currently played highest card = {self.target_meld()}')

        # Note: May be BLOCKING!!! if it waits for the player to play
        # Really it should yield to the main thread while the player thinks about it
        action = player.play()
        if action == '␆':
            # This is the player thinking (noop)
            return
        # Check if it's valid
        if self.target_meld() and action.cards and action < self.target_meld():
            # Punish the player for cheating
            raise

        self.notify_listeners("notify_play", player, action)
        if not action.cards:
            # That's a pass - Inactive for this hand
            self.active_players.remove(player)
        else:
            # Accept the action, and execute
            for card in action.cards:
                player._hand.remove(card)
                self.discards.append(card)
            # Remember the new highest meld from that player
            self.current_melds[self.players.index(player)] = action
            logging.debug(f'{player.name} is left with {player}')
            # Did they play out?
            if player.report_remaining_cards() == 0:
                self.set_player_finished(player)
            # Get the next player by moving the active player to the end of the queue
            self.active_players.append(self.active_players.pop(0))

    def play(self):
        """
        Play an episode with the given deck. All players get their reward
        Use the callback to inform of progress
        return current rankings
        """
        if self.state == State.INITIALISED:
            # Do an episode - We need 4 players and a deck of cards.
            shuffle(self.deck)
            self.state = State.DEALING
            self.deal()
        elif self.state == State.DEALING:
            while self.state == State.DEALING:
                # TODO: Time delay the dealing, and notify callbacks?
                self.state = State.SWAPPING
            # init the swapping state
            # Swap cards and decide who starts
            for player in self.players:
                logging.debug(f'{player.name} has {player}')
            self.swap_cards()
        elif self.state == State.SWAPPING:
            while self.state == State.SWAPPING:
                # TODO: Time delay the swapping, and notify callbacks?
                self.state = State.ROUND_STARTING
            self.pick_round_starter()
            self.positions = []
        elif self.state == State.ROUND_STARTING:
            # All privileges have been actioned, so reset positions and play until they are established anew
            self.active_players = self.players_with_cards()
            self.current_melds = ['␆', '␆', '␆', '␆']
            self.state = State.PLAYING
        elif self.state == State.PLAYING:
            # Play until the round is won (only one player remaining)
            self.player_turn()
            if len(self.active_players) == 1:
                self.state = State.HAND_WON
        elif self.state == State.HAND_WON:
            if len(self.positions) == 4:
                self.post_episode_checks()
                self.state = State.FINISHED
            else:
                assert (len(self.active_players) == 1)
                self.notify_listeners("notify_hand_won", self.active_players[0])
                # Winner gets to start the next round
                self.move_to_front(self.active_players[0])
                self.state = State.ROUND_STARTING

        return self.positions

    # State is a list of INT16. Each player is 7 ints, which are a bit-mask for:
    # The Hand (54 bits), the state (2 bits), the position (4 bits) and the meld (54 bits)
    def save_state(self):
        """"Encode each hand, each (current) meld, state and positions for each player"""
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
        # TODO in any case serialization should be in the episode
        deserialized_a = pickle.loads(serialized)
        # In any case, restore the gm to a blank state
        self.clear()
        # TODO: Dear god fix this.
        # first comes names, then class names, then the hand nad meld
        player_names = deserialized_a[0]
        player_types = deserialized_a[1]
        game_state = deserialized_a[2]
        for i, name in enumerate(player_names):
            # TODO: Nasty - turn the name into a class
            player_class = eval(player_types[i])
            player = self.make_player(player_class, name)
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
        # print(f'Serialized as {state}')
        self.restore_state(state)
