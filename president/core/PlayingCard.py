#!/usr/bin/env python
# -*- coding: utf-8 -*-
from termcolor import colored


class PlayingCard:
    suit_list = ["♠", "♣", "♦", "♥"]
    rank_list = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "Joker"]

    def __init__(self, index):
        num_suits = len(PlayingCard.suit_list)
        self._index = index
        self._value = index // num_suits
        self._suit = index % num_suits

    def __lt__(self, other):
        try:
            return self._value < other.get_value()
        except AttributeError:
            # I don't know how to compare to other
            return NotImplemented

    def __eq__(self, other):
        try:
            return self._value == other.get_value()
        except AttributeError:
            # I don't know how to compare to other
            return NotImplemented

    def __hash__(self):
        return self._value

    def same_card(self, other):
        """Exact match: same rank AND suit."""
        try:
            return self._index == other.get_index()
        except AttributeError:
            return NotImplemented

    def __le__(self, other):
        try:
            return self._value <= other.get_value()
        except AttributeError:
            # I don't know how to compare to other
            return NotImplemented

    def __str__(self):
        card_str = self.rank_str()
        # Most cards get the suit added (not Joker)
        if self._value < (len(PlayingCard.rank_list) - 1):
            card_str += self.suit_str()
        if self.isRed():
            return colored(card_str, 'red')
        return card_str

    def isRed(self):
        return self._suit in (2, 3) or (self._value == 13 and self._suit == 1)

    def get_index(self):
        return self._index

    def get_value(self):
        return self._value

    def get_suit(self):
        return self._suit

    def suit_str(self):
        return PlayingCard.suit_list[self._suit]

    def rank_str(self):
        return PlayingCard.rank_list[self._value]