#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Human player is a Player with console output (overloading the callbacks)
The Play function presents a list of the possible plays, and lets the user decide the move
"""
import logging
from asshole.player.AbstractPlayer import AbstractPlayer, possible_plays


class HumanPlayer(AbstractPlayer):
    def __init__(self, name):
        AbstractPlayer.__init__(self, name)
        self.opp_status = ['Waiting', 'Waiting', 'Waiting']

    def surrender_cards(self, cards, receiver):
        # Call super function, but also report on the interaction
        print("Must give {} to {}".format(", ".join(c.__str__() for c in cards), receiver.name))
        super().surrender_cards(cards, receiver)

    def award_cards(self, cards, giver):
        # Call super function, but also report on the interaction
        print("Just got {} from {}".format(", ".join(c.__str__() for c in cards), giver.name))
        super().award_cards(cards, giver)

    # Special card_game_listener behaviour
    def notify_hand_start(self, starter):
        super(HumanPlayer, self).notify_hand_start(starter)
        if starter == self:
            print("YOU get to start the round")
        else:
            print("{} gets to start the round".format(starter.name))

    def notify_hand_won(self, winner):
        super(HumanPlayer, self).notify_hand_won(winner)
        print("--{} wins the round--".format(winner.name))

    def notify_played_out(self, player, pos):
        super(HumanPlayer, self).notify_played_out(player, pos)
        if player == self:
            print(f'You ({player.name}) played out, and is ranked {self.ranking_names[pos]}')
        else:
            print(f'{player.name} played out, and is ranked {self.ranking_names[pos]}')
            self.opp_status[self.opponents.index(player)] = self.ranking_names[pos]

    def notify_play(self, player, meld):
        super(HumanPlayer, self).notify_play(player, meld)
        # TODO: for some reason the we get notified before the cards are taken from the hand
        # TODO: len(meld) should work, not need len(meld.cards)
        if player != self:
            print(f'{player.name} plays {meld}, leaving {len(player._hand)- len(meld.cards1)} cards')
            self.opp_status[self.opponents.index(player)] = meld

    def show_player(self, index):
        """Report on the last card played, and the total cards of the opposition @ given index"""
        opp = self.opponents[index]
        tabs = "\t" * ((index - 1) * (3 * index - 4))
        print("{}{} has {} cards".format(tabs, opp.name, opp.report_remaining_cards()))
        # TODO: Remember other player's status
        print(f"{tabs}\t{self.opp_status[index]}")

    def play(self):
        """Must return a meld (set of cards). Pass is an empty set"""
        # Show the table
        print("-" * 20)
        for i in [2, 1, 0]:
            self.show_player(i)
        print(self)

        selection = possible_plays(self._hand, self.target_meld, self.name)

        card_selection_string = "Select card: \n"
        for key, value in enumerate(selection):
            # meld_string = " & ".join([c.__str__() for c in value.cards])
            # Check to see if this play will be a split (ignore 2s and Js)
            split_string = ""
            if len(value.cards) > 0 and self.will_split(value):
                split_string = " - split"
            card_selection_string += "[{}] : {}{}\n".format(key, value, split_string)

        # INPUT HERE
        valid_input = False
        card_index = -1
        user_input = input(card_selection_string)
        while not valid_input:
            try:
                if not user_input:
                    card_index = 0
                else:
                    card_index = int(user_input)
                if card_index < len(selection):
                    valid_input = True
                else:
                    user_input = "<error>"
            except IndexError:
                user_input = input("Oops. Enter number from 0 to {}".format(len(selection) - 1))
            except ValueError:
                user_input = input("Oops. Enter number (from 0 to {})".format(len(selection) - 1))

        logging.info("{} tries to play option {} which is a {}".format(self.name, card_index, selection[card_index]))
        return selection[card_index]
