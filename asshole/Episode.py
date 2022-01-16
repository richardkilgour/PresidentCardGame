#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Play a single episode
"""
import logging


class Episode:
    def __init__(self, players, positions, deck, listener_list):
        # All the played cards (for checking)
        self.discards = []
        # List of players in their positions from last round
        # Index 0 is King, Index 3 is Asshole
        # None if it's a new game and no positions have been established
        # The Episode will (re) populate this and return after the episode has run
        self.positions = positions
        # The deck to use (should be shuffled already)
        self.deck = deck
        # List of the players
        self.players = players
        # The highest current play
        self.target_meld = None
        # All the listeners
        self.listener_list = listener_list

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
                if card.value() > target_card_value:
                    break
                if card.value() == target_card_value and + card.suit() == target_card_suit:
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

    def play_one_round(self):
        """
        Play one entire round, from initial meld to winning meld
        Everyone plays in order of self.player
        Return the winner.
        Even the Asshole "wins" the last round, and their cards are drained
        """
        # Everyone who has cards is now active
        # This is a circular list:
        #   Players who pass
        #   Players that play are moved to the back
        # Round is over when one player remains active
        active_players = self.players_with_cards()

        logging.info(f"Players with cards remaining = {active_players}")
        # ================
        # Round Starts with some poor, lonely sod.
        # Send the hand finished code!!!
        # ================
        if len(active_players) == 1:
            self.set_player_finished(active_players[0])

        self.target_meld = None

        # Play until we have a winner (everyone else has passed)
        while len(active_players) > 1:
            player = active_players[0]
            if player.report_remaining_cards() == 0:
                # This payer has played out, but others have played on their last card
                active_players.remove(player)
                continue

            logging.info("Currently played highest card = {}".format(self.target_meld))

            # =======================
            # LOCK ON SNAPSHOTS
            # Snapshottting here has undefined consequences
            # player state and order are in flux. wait for the meld or other decision
            # =======================
            action = player.play()
            # Check if it's valid
            if self.target_meld and action.cards and action < self.target_meld:
                # Punish the player for cheating
                raise
            self.notify_listeners("notify_play", player, action)
            if not action.cards:
                # That's a pass - Inactive for this hand
                active_players.remove(player)
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
                active_players.append(active_players.pop(0))
            # =======================
            # UNLOCK ON SNAPSHOTS
            # Snapshottting hereforth is OK
            # =======================

        self.notify_listeners("notify_hand_won", active_players[0])
        # Winner gets to start the next round
        self.move_to_front(active_players[0])

    def play_episode(self):
        """
        Play an episode with the given deck. All players get their reward
        Use the callback to inform of progress
        return new rankings
        """
        self.deal()

        for player in self.players:
            logging.debug("{} has {}".format(player.name, player))

        # Swap cards and decide who starts
        self.swap_cards()
        self.pick_round_starter()

        # All privileges have been actioned, so reset positions and play until they are established anew
        self.positions = []

        while len(self.positions) < 4:
            self.play_one_round()

        self.post_episode_checks()

        return self.positions
