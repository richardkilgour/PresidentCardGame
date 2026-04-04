#!/usr/bin/env python
# -*- coding: utf-8 -*-
from functools import total_ordering
from president.core.PlayingCard import PlayingCard


@total_ordering # gives __lt__, __ge__, __ne__ for free
class Meld:
    """
    Meld is a glorified list of cards
    All are the same value, but there can be up to 4 of them
    They should be in ascending order based on card._index
    """
    def __init__(self, card=None, meld=None):
        self.cards = []
        if card:
            self.cards = [card]
        if meld:
            # Sort ascending by card index
            self.cards = sorted(meld.cards + [card], key=lambda c: c.get_index())

    def __len__(self):
        return len(self.cards)

    def __eq__(self, other):
        if not isinstance(other, Meld):
            return NotImplemented
        return self.cards == other.cards


    def __gt__(self, other):
        """Returns True if self beats other in play."""
        if not isinstance(other, Meld):
            return NotImplemented

        # Empty meld (pass) loses to everything
        if not self.cards:
            return False

        # Everything beats a pass
        if not other.cards:
            return True

        # self has higher face value — now check length rules
        self_len = len(self)
        other_len = len(other)
        self_val = self.cards[0].get_value()
        other_val = other.cards[0].get_value()

        # Lower or equal face value cannot win
        if self_val <= other_val:
            return False
        elif other_len == 1:
            # If target length is 1, then any higher single will win
            # Double's can't beat a single
            return self_len == 1
        elif other_len == 2:
            if self_val >= 12:
                # A single 2 or Joker beats any pair
                return self_len == 1
            # Beaten by a higher pair
            return self_len == 2
        elif other_len == 3:
            if self_val == 12:
                # Double 2 beats a triple
                return self_len == 2
            if self_val == 13:
                if other_val == 12:
                    # 2 Jokers are needed to beat a triple 2
                    return self_len == 2
                # 1 Joker beats any other triple
                return self_len == 1
            # Three higher cards to beat three cards
            return self_len == 3
        elif other_len == 4:
            # Special cases for 2 and joker
            if self_val == 12:
                # Triple 2 beats a quad
                return self_len == 3
            if self_val == 13:
                if other_val == 12:
                    # Cannot beat quad 2s with anything - not even 2 Jokers!
                    return False
                # Double Joker wins
                return self_len == 2
            return self_len == 4

    def __le__(self, other):
        """Returns True if self does not beat other."""
        if not isinstance(other, Meld):
            return NotImplemented
        return not (self > other)

    def __str__(self):
        if not self.cards:
            return "<pass>"
        return "[" + " & ".join([c.__str__() for c in self.cards]) + "]"


def test_cases():
    # Some unit tests for meld
    single_3 = Meld(PlayingCard(0))
    assert (single_3 <= single_3)

    double_3 = Meld(PlayingCard(3), single_3)
    # Singles can't beat doubles; doubles can't beat singles
    assert (double_3 <= single_3)
    assert (single_3 <= double_3)

    single_4 = Meld(PlayingCard(4))
    assert (single_4 > single_3)
    assert (single_3 <= single_4)
    assert (single_4 <= single_4)
    assert (double_3 <= single_4)
    # Singles can't beat doubles; doubles can't beat singles
    assert (single_4 <= double_3)

    double_4 = Meld(PlayingCard(5), single_4)
    assert (double_4 > double_3)
    # Singles can't beat doubles; doubles can't beat singles
    assert (double_4 <= single_3)
    assert (double_4 <= single_4)

    single_2 = Meld(PlayingCard(49))
    # 2s are the second best card
    assert (single_3 <= single_2)
    assert (single_4 <= single_2)
    # Singles can't beat doubles; doubles can't beat singles
    assert (double_3 <= single_2)
    assert (double_4 <= single_2)

    triple_4 = Meld(PlayingCard(6), double_4)
    # Singles can't beat doubles; doubles can't beat singles
    assert (single_2 <= triple_4)

    triple_3 = Meld(PlayingCard(0), Meld(PlayingCard(1), Meld((PlayingCard(2)))))
    triple_8 = Meld(PlayingCard(24), Meld(PlayingCard(25), Meld((PlayingCard(26)))))
    # Triple 8 should beat triple 3
    assert (triple_8 > triple_3)

    # Special case for 2s and jokers - double 2 should beat triple 3
    double_2 = Meld(PlayingCard(50), single_2)
    assert (double_2 > triple_3)
    assert (single_2 <= triple_3)


if __name__ == '__main__':
    test_cases()
    print("All tests passed")
