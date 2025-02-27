#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Human players is a Player with console output (overloading the callbacks)
The Play function presents a list of the possible plays, and lets the user decide the move
"""
import logging
from asshole.core.AbstractPlayer import AbstractPlayer, possible_plays


class ConsolePlayer(AbstractPlayer):
    def __init__(self, name):
        AbstractPlayer.__init__(self, name)

    def surrender_cards(self, cards, receiver):
        # Call super function, but also report on the interaction
        print(f'Must give {", ".join(c.__str__() for c in cards)} to {receiver.name}')
        super().surrender_cards(cards, receiver)

    def award_cards(self, cards, giver):
        # Call super function, but also report on the interaction
        print(f'Just got {", ".join(c.__str__() for c in cards)} from {giver.name}')
        super().award_cards(cards, giver)

    # Special card_game_listener behaviour
    def notify_hand_start(self, starter):
        super(ConsolePlayer, self).notify_hand_start(starter)
        if starter == self:
            print("YOU get to start the round")
        else:
            print(f'{starter.name} gets to start the round')

    def notify_hand_won(self, winner):
        super(ConsolePlayer, self).notify_hand_won(winner)
        print(f'--{winner.name} wins the round--')

    def notify_played_out(self, player, pos):
        super(ConsolePlayer, self).notify_played_out(player, pos)
        if player == self:
            print(f'You ({player.name}) played out, and is ranked {self.ranking_names[pos]}')
        else:
            print(f'{player.name} played out, and is ranked {self.ranking_names[pos]}')
            self.player_status[self.players.index(player)] = self.ranking_names[pos]

    def notify_play(self, player, meld):
        super(ConsolePlayer, self).notify_play(player, meld)
        # TODO: for some reason the we get notified before the cards are taken from the hand
        if player != self:
            print(f'{player.name} plays {meld}, leaving {len(player._hand)- len(meld.cards)} cards')
            self.player_status[self.players.index(player)] = meld

    def show_player(self, index):
        """Report on the last card played, and the total cards of the opposition @ given index"""
        opp = self.players[index]
        tabs = "\t" * ((index - 1) * (3 * index - 4))
        print(f'{tabs}{opp.name} has {opp.report_remaining_cards()} cards')
        print(f'{tabs}\t{self.player_status[index]}')

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
            card_selection_string += f'[{key}] : {value}{split_string}\n'

        # INPUT HERE (Blocking)
        user_input = input(card_selection_string)
        # Send noop on input error, and repeat next loop
        try:
            # None just return option 0
            if not user_input:
                card_index = 0
            else:
                card_index = int(user_input)
            if card_index > len(selection):
                return '␆'
        except IndexError:
            print(f'Oops. Enter number from _0 to {len(selection) - 1}_')
            return '␆'
        except ValueError:
            print(f'Oops. Enter _number_ (from 0 to {len(selection) - 1}')
            return '␆'

        logging.info(f'{self.name} tries to play option {card_index} which is a {selection[card_index]}')
        return selection[card_index]
