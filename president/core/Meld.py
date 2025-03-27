#!/usr/bin/env python
# -*- coding: utf-8 -*-
from president.core.PlayingCard import PlayingCard


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
            self.cards = meld.cards + [card]
            # Sort by card index
            self.cards.sort(key=lambda c: c.get_index())

    def __gt__(self, other):
        """self <= other"""
        # Empty is lowest
        if not self.cards:
            return False
        # Everything beats a pass
        if not other.cards:
            return True

        # In all cases, face value lower or equal is lower
        if self.cards[0] <= other.cards[0]:
            return False

        # We know that self has a higher face value, so need to check the lengths

        # If target length is 1, then any higher single will win
        if len(other.cards) == 1:
            # Double's can't beat a single
            return len(self.cards) == 1
        elif len(other.cards) == 2:
            # Special case for 3 and joker
            if self.cards[0].get_value() >= 12:
                return len(self.cards) == 1
            return len(self.cards) == 2
        elif len(other.cards) == 3:
            # Special case for 3 and joker
            if self.cards[0].get_value() == 12:
                return len(self.cards) == 2
            if self.cards[0].get_value() == 13:
                if other.cards[0] == 12:
                    # Triple 2 needs 2 Jokers
                    return len(self.cards) == 2
                # Otherwise one Joker will be enough
                return len(self.cards) == 1
            # Three cards to beat three cards
            return len(self.cards) == 3
        elif len(other.cards) == 4:
            # Special case for 3 and joker
            if self.cards[0].get_value() == 12:
                return len(self.cards) == 3
            if self.cards[0].get_value() == 13:
                if other.cards[0] == 12:
                    # Three jokers? Cheat!!!
                    assert False
                return len(self.cards) == 2
            return len(self.cards) == 4

    def __le__(self, other):
        # self <= other
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
