#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TODO: define responsibilities (this was designed for Reienforcement LEarning)
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
        # The deck to use (should be shuffled already?)
        # TODO: decide who owns the deck (and thier responsibilities)
        self.deck = deck
        # List of the players
        self.players = players
        # The highest current play
        self.target_meld = None
        # All the listeners
        self.listener_list = listener_list
        self.state = State.INITIALISED
        self.active_players = []

    def swap_cards(self):
        """Swap cards if necessary"""
        if self.positions:
            # Asshole swaps 2 cards with King
            print("{} must give {} 2 cards".format(self.positions[3].name, self.positions[0].name))
            tribute = [self.positions[3]._hand[-2], self.positions[3]._hand[-1]]
            self.positions[3].surrender_cards(tribute, self.positions[0])
            discard = [self.positions[0]._hand[0], self.positions[0]._hand[1]]
            self.positions[0].surrender_cards(discard, self.positions[3])
            logging.debug(
                "{} swapped {} and {} for {} cards {} and {}".format(self.positions[3].name, tribute[0], tribute[1],
                                                                     self.positions[0].name, discard[0], discard[1]))
            # Prince swaps 1 card with Citizen
            print("{} must give {} 1 card".format(self.positions[2].name, self.positions[1].name))
            tribute = [self.positions[2]._hand[-1]]
            self.positions[2].surrender_cards(tribute, self.positions[1])
            discard = [self.positions[1]._hand[0]]
            self.positions[1].surrender_cards(discard, self.positions[2])
            logging.debug(
                "{} swapped {} for {} cards {}".format(self.positions[2].name, tribute[0], self.positions[1].name,
                                                       discard[0]))
            for player in self.players:
                # Log everyone's hands after teh swap
                logging.debug("{} has {}".format(player.name, player))

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
        logging.info("{} is the Asshole!!! Left with {}".format(asshole.name, asshole._hand))
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

        logging.info("Currently played highest card = {}".format(self.target_meld))

        # Note: May be BLOCKING!!! if it waits for the player to play
        # Really it should yield to the main thread while the player thinks about it
        action = player.play()
        if action == '␆':
            # This is the player thinking (noop)
            return
        # Check if it's valid
        if self.target_meld and action.cards and action < self.target_meld:
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
            # Remember the new highest meld
            self.target_meld = action
            logging.debug("{} is left with {}".format(player.name, player))
            # Did they play out?
            if player.report_remaining_cards() == 0:
                self.set_player_finished(player)
            # Get the next player by moving the active player to the end of the queue
            self.active_players.append(self.active_players.pop(0))

    def play(self):
        """
        Play an episode with the given deck. All players get their reward
        Use the callback to inform of progress
        return new rankings
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
                logging.debug("{} has {}".format(player.name, player))
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
            self.target_meld = None
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
                assert(len(self.active_players)==1)
                self.notify_listeners("notify_hand_won", self.active_players[0])
                # Winner gets to start the next round
                self.move_to_front(self.active_players[0])
                self.state = State.ROUND_STARTING

        return self.positions
