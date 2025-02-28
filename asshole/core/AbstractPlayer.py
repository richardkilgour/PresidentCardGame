#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from abc import abstractmethod
import numpy as np

from asshole.core.Meld import Meld
from asshole.core.CardGameListener import CardGameListener
from asshole.core.PlayingCard import PlayingCard

class AbstractPlayer(CardGameListener):
    # the listener keeps track of all cards played
    ranking_names = ["King", "Prince", "Citizen", "Asshole"]

    def __init__(self, name):
        super().__init__()
        self.name = name
        self._hand = []
        # TODO: ask the GM?
        self.target_meld = None
        self.position_count = [0, 0, 0, 0]
        self.last_played = None

    def set_position(self, pos):
        # Remember the new position for statistical porpoises
        self.position_count[pos] += 1

    def get_score(self):
        return 2 * self.position_count[0] + \
            self.position_count[1] - \
            self.position_count[2] - \
            2 * self.position_count[3]

    def card_to_hand(self, card):
        self._hand.append(card)
        self._hand.sort(key=lambda c: c.get_index())

    def get_hand_indices(self):
        # Useful for exact matching
        return [c.get_index() for c in self._hand]

    def notify_hand_won(self, winner):
        # Someone just won the hand
        super().notify_hand_won(winner)
        self.target_meld = None

    def notify_play(self, player, meld):
        # Ignore passes
        if not self.target_meld or meld > self.target_meld:
            self.target_meld = meld

    def possible_plays(self, target_meld):
        """
        Make a list of possible melds that may be played given the target minimum meld
        Will always return the pass option (in last place)
        Example return value: [[3], [3, 3], [4], ..., []]
        """
        # A list of the melds possible to play
        possible_melds = []

        if not target_meld:
            # If no minimum, all melds are playable
            for card in self._hand:
                # Check to see if this can make a pair
                if len(possible_melds) > 0 and card == possible_melds[-1].cards[0]:
                    possible_melds.append(Meld(card, possible_melds[-1]))
                else:
                    possible_melds.append(Meld(card))
        else:
            current_meld = None
            for card in self._hand:
                if current_meld and (card.get_value() == current_meld.cards[0].get_value()):
                    current_meld = Meld(card, current_meld)
                else:
                    current_meld = Meld(card)
                if current_meld > target_meld:
                    possible_melds.append(current_meld)

        # The option for Pass
        possible_melds.append(Meld())

        card_string = f"{self.name} has:"
        for s in self._hand:
            card_string += " {},".format(s)
        logging.debug(card_string)

        meld_string = f"Options for {self.name} to play are:"
        for s in possible_melds:
            meld_string += " {},".format(s)
        logging.debug(meld_string)

        return possible_melds

    @abstractmethod
    def play(self):
        # Must be implemented by children - BLOCKING (Don't expect any redraws)
        pass

    def report_remaining_cards(self):
        return len(self._hand)

    def number_of_cards_of_value(self, value):
        """
        How many card in the hand of the given value
        Utility to check for pairs etc
        """
        count = 0
        for c in self._hand:
            if c.get_value() == value:
                count += 1
        return count

    def surrender_cards(self, cards, receiver):
        for c in cards:
            self._hand.remove(c)
        receiver.award_cards(cards, self)

    def award_cards(self, cards, giver):
        self._hand += cards
        self._hand.sort()

    def __str__(self):
        out = ""
        for card in self._hand:
            out += card.__str__() + " "
        return out

    # Inform the players if a given meld will split the cards
    def will_split(self, candidate):
        count_of_value = 0
        # Find the first card of the same value
        for c in self._hand:
            if c.get_value() == candidate.cards[0].get_value():
                # Count the number of cards of the same value
                count_of_value += 1
        return len(candidate.cards) < count_of_value

    @staticmethod
    def encode_hand(hand):
        """Encode the hand as a numpy array"""
        h = np.zeros((4, 14), dtype=int)
        for c in hand:
            h[c.get_suit(), c.get_value()] = 1
        return h

    @staticmethod
    def decode_hand(encoding):
        """Decode a numpy array as a hand"""
        assert (encoding.shape == (4, 14))
        hand = []
        for index, x in np.ndenumerate(encoding):
            if x == 1:
                hand.append(PlayingCard(index[0] + index[1] * 4))
        hand.sort()
        return hand

    @staticmethod
    def encode_meld(meld):
        """Encode the meld as a numpy array"""
        h = np.zeros((4, 14), dtype=int)
        for c in meld.cards:
            h[c.get_suit(), c.get_value()] = 1
        return h

    def encode(self):
        hand = self.encode_hand(self._hand)

        # TODO: Status is now tracked by the GM (or episode)
        status = self.get_status()
        if status == "passed":
            meld = self.encode_hand([])
            hand[2, 13] = 1
        elif status == "waiting":
            meld = self.encode_hand([])
            hand[3, 13] = 1
        else:
            meld = self.encode_meld(status)

        return np.concatenate((hand, meld), axis=1)


def main():
    # Test for hand encoding (static functions only)
    test_hand = []
    for i in range(0, 54):
        test_hand.append(PlayingCard(i))
        for j, h in enumerate(AbstractPlayer.decode_hand(AbstractPlayer.encode_hand(test_hand))):
            assert (h == test_hand[j])
        test_card = PlayingCard(i)
        assert AbstractPlayer.decode_hand(AbstractPlayer.encode_hand([test_card]))[0] == test_card
    # Make a new players without a GM
    player = AbstractPlayer('')
    # Give it some cards - a single card of each type
    for i in range(0, 54, 4):
        player.card_to_hand(PlayingCard(i))
    # Check the possible melds for each single card
    for i in range(0, 54, 4):
        # Any single meld should result in possible plays inversely proportional to card value
        meld = Meld(PlayingCard(i))
        player.notify_play(None, meld)
        assert (len(player.possible_plays(meld)) == 14 - i // 4)
        # Any double plays can't be responded to except by 2, Joker, or pass
        meld = Meld(PlayingCard(i + 1), meld)
        player.notify_play(None, meld)
        if meld.cards[0].get_value() < 12:
            assert (len(player.possible_plays(meld)) == 3)
        elif meld.cards[0].get_value() == 12:
            # Double 2: Joker of pass
            assert (len(player.possible_plays(meld)) == 2)
        else:
            # Double Joker: only Pass
            assert (len(player.possible_plays(meld)) == 1)

    # Test case based on some bug
    player = AbstractPlayer('')
    # 3♦ 5♥ 5♠ 6♠ 7♦ 8♦ 9♥ 9♠ J♣ J♦ 2♦ Joker
    for card in ([2, 9, 8, 12, 18, 28, 25, 24, 35, 34, 50, 52]):
        player.card_to_hand(PlayingCard(card))
    meld =  Meld(PlayingCard(53))
    player.notify_play(None, meld)
    print(player.possible_plays(meld))


if __name__ == '__main__':
    main()
