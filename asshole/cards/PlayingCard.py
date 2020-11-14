#!/usr/bin/env python
# -*- coding: utf-8 -*-
from termcolor import colored


class PlayingCard:
    suit_str = ["♠", "♥", "♦", "♣"]
    rank_str = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "Joker"]

    def __init__(self, index):
        num_suits = len(PlayingCard.suit_str)
        self._value = index // num_suits
        self._suit = index % num_suits

    def __lt__(self, other):
        try:
            return self._value < other.value()
        except AttributeError:
            # I don't know how to compare to other
            return NotImplemented

    def __eq__(self, other):
        try:
            return self._value == other.value()
        except AttributeError:
            # I don't know how to compare to other
            return NotImplemented

    def __le__(self, other):
        try:
            return self._value <= other.value()
        except AttributeError:
            # I don't know how to compare to other
            return NotImplemented

    def __str__(self):
        card_str = PlayingCard.rank_str[self._value]
        # Most cards get the suit added (not Joker)
        if self._value < (len(PlayingCard.rank_str) - 1):
            card_str += PlayingCard.suit_str[self._suit]
        if self._suit == 1 or self._suit == 2:
            return colored(card_str, 'red')
        return card_str

    def value(self):
        return self._value

    def suit(self):
        return self._suit