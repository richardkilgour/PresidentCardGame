#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for the Meld class.

Tests cover:
- Basic meld creation and properties
- Comparison operators (>, <=, ==)
- Edge cases (empty melds, single cards, multiples)
- Special card rules (2s and Jokers)
- String representation
"""
import unittest
from president.core.Meld import Meld
from president.core.PlayingCard import PlayingCard


class TestMeldCreation(unittest.TestCase):
    """Test meld creation and basic properties."""

    def test_empty_meld(self):
        """Empty meld represents a pass."""
        meld = Meld()
        self.assertEqual(len(meld), 0)
        self.assertEqual(meld.cards, [])
        self.assertEqual(str(meld), "<pass>")

    def test_single_card_meld(self):
        """Single card meld creation."""
        card = PlayingCard(0)  # 3♠
        meld = Meld(card)
        self.assertEqual(len(meld), 1)
        self.assertEqual(meld.cards[0], card)

    def test_double_meld(self):
        """Double meld creation and card ordering."""
        card1 = PlayingCard(0)  # 3♠
        card2 = PlayingCard(1)  # 3♣
        single = Meld(card1)
        double = Meld(card2, single)
        
        self.assertEqual(len(double), 2)
        self.assertEqual(double.cards[0], card1)  # Cards sorted by index
        self.assertEqual(double.cards[1], card2)

    def test_triple_meld(self):
        """Triple meld creation."""
        card1 = PlayingCard(0)  # 3♠
        card2 = PlayingCard(1)  # 3♣
        card3 = PlayingCard(2)  # 3♦
        single = Meld(card1)
        double = Meld(card2, single)
        triple = Meld(card3, double)
        
        self.assertEqual(len(triple), 3)
        self.assertEqual(triple.cards[0], card1)
        self.assertEqual(triple.cards[2], card3)

    def test_quad_meld(self):
        """Quad meld creation."""
        cards = [PlayingCard(i) for i in range(4)]  # All 3s
        meld = None
        for card in cards:
            meld = Meld(card, meld) if meld else Meld(card)
        
        self.assertEqual(len(meld), 4)
        self.assertEqual(len(meld.cards), 4)

    def test_meld_card_sorting(self):
        """Cards should be sorted by index regardless of insertion order."""
        card1 = PlayingCard(2)  # 3♦
        card2 = PlayingCard(0)  # 3♠
        card3 = PlayingCard(1)  # 3♣
        
        single = Meld(card1)
        double = Meld(card2, single)
        triple = Meld(card3, double)
        
        # Should be sorted: 0, 1, 2
        self.assertEqual(triple.cards[0].get_index(), 0)
        self.assertEqual(triple.cards[1].get_index(), 1)
        self.assertEqual(triple.cards[2].get_index(), 2)


class TestMeldEquality(unittest.TestCase):
    """Test meld equality comparisons."""

    def test_empty_melds_equal(self):
        """Two empty melds are equal."""
        meld1 = Meld()
        meld2 = Meld()
        self.assertEqual(meld1, meld2)

    def test_single_card_equality(self):
        """Single card melds are equal if same card."""
        card1 = PlayingCard(4)  # 4♠
        card2 = PlayingCard(4)  # 4♠
        meld1 = Meld(card1)
        meld2 = Meld(card2)
        self.assertEqual(meld1, meld2)

    def test_single_card_inequality(self):
        """Single card melds are unequal if different cards."""
        card1 = PlayingCard(4)  # 4♠
        card2 = PlayingCard(5)  # 4♣
        meld1 = Meld(card1)
        meld2 = Meld(card2)
        self.assertNotEqual(meld1, meld2)

    def test_double_equality(self):
        """Double melds are equal if same cards."""
        meld1 = Meld(PlayingCard(1), Meld(PlayingCard(0)))
        meld2 = Meld(PlayingCard(1), Meld(PlayingCard(0)))
        self.assertEqual(meld1, meld2)

    def test_different_lengths_unequal(self):
        """Melds of different lengths are unequal."""
        single = Meld(PlayingCard(0))
        double = Meld(PlayingCard(1), Meld(PlayingCard(0)))
        self.assertNotEqual(single, double)


class TestMeldComparison(unittest.TestCase):
    """Test meld comparison logic (> operator)."""

    def test_pass_loses_to_everything(self):
        """Empty meld (pass) loses to any card."""
        pass_meld = Meld()
        single_3 = Meld(PlayingCard(0))
        
        self.assertFalse(pass_meld > single_3)
        self.assertTrue(single_3 > pass_meld)

    def test_higher_single_beats_lower_single(self):
        """Higher value single card beats lower value."""
        single_3 = Meld(PlayingCard(0))   # 3♠
        single_4 = Meld(PlayingCard(4))   # 4♠
        single_5 = Meld(PlayingCard(8))   # 5♠
        
        self.assertTrue(single_4 > single_3)
        self.assertTrue(single_5 > single_4)
        self.assertFalse(single_3 > single_4)

    def test_single_cannot_beat_double(self):
        """Single card cannot beat a double (except special rules)."""
        single_10 = Meld(PlayingCard(28))  # 10♠
        double_3 = Meld(PlayingCard(1), Meld(PlayingCard(0)))  # Double 3
        
        self.assertFalse(single_10 > double_3)

    def test_double_cannot_beat_single(self):
        """Double cannot beat a single."""
        single_3 = Meld(PlayingCard(0))
        double_4 = Meld(PlayingCard(5), Meld(PlayingCard(4)))
        
        self.assertFalse(double_4 > single_3)

    def test_higher_double_beats_lower_double(self):
        """Higher value double beats lower value double."""
        double_3 = Meld(PlayingCard(1), Meld(PlayingCard(0)))
        double_4 = Meld(PlayingCard(5), Meld(PlayingCard(4)))
        
        self.assertTrue(double_4 > double_3)
        self.assertFalse(double_3 > double_4)

    def test_triple_comparisons(self):
        """Triple meld comparisons."""
        triple_3 = Meld(PlayingCard(2), Meld(PlayingCard(1), Meld(PlayingCard(0))))
        triple_8 = Meld(PlayingCard(26), Meld(PlayingCard(25), Meld(PlayingCard(24))))
        
        self.assertTrue(triple_8 > triple_3)
        self.assertFalse(triple_3 > triple_8)

    def test_quad_comparisons(self):
        """Quad meld comparisons."""
        quad_3 = Meld(PlayingCard(3), Meld(PlayingCard(2), 
                 Meld(PlayingCard(1), Meld(PlayingCard(0)))))
        quad_5 = Meld(PlayingCard(11), Meld(PlayingCard(10), 
                 Meld(PlayingCard(9), Meld(PlayingCard(8)))))
        
        self.assertTrue(quad_5 > quad_3)
        self.assertFalse(quad_3 > quad_5)


class TestSpecialCardRules(unittest.TestCase):
    """Test special rules for 2s and Jokers."""

    def test_single_2_beats_any_double(self):
        """Single 2 beats any pair."""
        single_2 = Meld(PlayingCard(48))  # 2♠
        double_ace = Meld(PlayingCard(45), Meld(PlayingCard(44)))  # Double Ace
        
        self.assertTrue(single_2 > double_ace)

    def test_single_joker_beats_any_double(self):
        """Single Joker beats any pair."""
        single_joker = Meld(PlayingCard(52))  # Joker
        double_ace = Meld(PlayingCard(45), Meld(PlayingCard(44)))
        
        self.assertTrue(single_joker > double_ace)

    def test_single_joker_beats_any_triple_except_triple_2(self):
        """Single Joker beats any triple except triple 2."""
        single_joker = Meld(PlayingCard(52))
        triple_5 = Meld(PlayingCard(10), Meld(PlayingCard(9), Meld(PlayingCard(8))))
        triple_2 = Meld(PlayingCard(50), Meld(PlayingCard(49), Meld(PlayingCard(48))))
        
        self.assertTrue(single_joker > triple_5)
        self.assertFalse(single_joker > triple_2)

    def test_double_2_beats_any_triple(self):
        """Double 2 beats any triple."""
        double_2 = Meld(PlayingCard(49), Meld(PlayingCard(48)))
        triple_ace = Meld(PlayingCard(46), Meld(PlayingCard(45), Meld(PlayingCard(44))))
        
        self.assertTrue(double_2 > triple_ace)

    def test_double_joker_beats_triple_except_triple_2(self):
        """Double Joker beats any triple except triple 2."""
        double_joker = Meld(PlayingCard(53), Meld(PlayingCard(52)))
        triple_ace = Meld(PlayingCard(46), Meld(PlayingCard(45), Meld(PlayingCard(44))))
        triple_2 = Meld(PlayingCard(50), Meld(PlayingCard(49), Meld(PlayingCard(48))))
        
        self.assertTrue(double_joker > triple_ace)
        # Two jokers should still beat triple 2
        self.assertTrue(double_joker > triple_2)

    def test_double_joker_beats_double_2(self):
        """Double Joker beats double 2."""
        double_joker = Meld(PlayingCard(53), Meld(PlayingCard(52)))
        double_2 = Meld(PlayingCard(49), Meld(PlayingCard(48)))
        
        self.assertTrue(double_joker > double_2)

    def test_triple_2_beats_quad(self):
        """Triple 2 beats any quad."""
        triple_2 = Meld(PlayingCard(50), Meld(PlayingCard(49), Meld(PlayingCard(48))))
        quad_ace = Meld(PlayingCard(47), Meld(PlayingCard(46), 
                   Meld(PlayingCard(45), Meld(PlayingCard(44)))))
        
        self.assertTrue(triple_2 > quad_ace)

    def test_double_joker_beats_quad_except_quad_2(self):
        """Double Joker beats any quad except quad 2."""
        double_joker = Meld(PlayingCard(53), Meld(PlayingCard(52)))
        quad_ace = Meld(PlayingCard(47), Meld(PlayingCard(46), 
                   Meld(PlayingCard(45), Meld(PlayingCard(44)))))
        
        self.assertTrue(double_joker > quad_ace)


class TestLessThanOrEqual(unittest.TestCase):
    """Test <= operator (derived from >)."""

    def test_equal_melds(self):
        """Equal melds satisfy <=."""
        meld1 = Meld(PlayingCard(4))
        meld2 = Meld(PlayingCard(4))
        
        self.assertTrue(meld1 <= meld2)
        self.assertTrue(meld2 <= meld1)

    def test_lower_value(self):
        """Lower value satisfies <=."""
        single_3 = Meld(PlayingCard(0))
        single_4 = Meld(PlayingCard(4))
        
        self.assertTrue(single_3 <= single_4)
        self.assertFalse(single_4 <= single_3)

    def test_pass_less_than_or_equal_anything(self):
        """Pass is <= any card."""
        pass_meld = Meld()
        single_3 = Meld(PlayingCard(0))
        
        self.assertTrue(pass_meld <= single_3)


class TestStringRepresentation(unittest.TestCase):
    """Test string representation of melds."""

    def test_empty_meld_string(self):
        """Empty meld displays as <pass>."""
        meld = Meld()
        self.assertEqual(str(meld), "<pass>")

    def test_single_card_string(self):
        """Single card meld string."""
        card = PlayingCard(0)  # 3♠
        meld = Meld(card)
        # Should contain the card representation
        meld_str = str(meld)
        self.assertIn("[", meld_str)
        self.assertIn("]", meld_str)

    def test_double_card_string(self):
        """Double card meld string."""
        double = Meld(PlayingCard(1), Meld(PlayingCard(0)))
        meld_str = str(double)
        # Should contain & separator for multiple cards
        self.assertIn("&", meld_str)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and potential issues."""

    def test_same_value_different_suits(self):
        """Cards of same value but different suits."""
        card1 = PlayingCard(0)  # 3♠
        card2 = PlayingCard(1)  # 3♣
        
        meld1 = Meld(card1)
        meld2 = Meld(card2)
        
        # Same value, so neither beats the other in single play
        self.assertFalse(meld1 > meld2)
        self.assertFalse(meld2 > meld1)

    def test_joker_indices(self):
        """Jokers have special indices (52, 53)."""
        joker1 = PlayingCard(52)  # Black Joker
        joker2 = PlayingCard(53)  # Red Joker
        
        self.assertEqual(joker1.get_value(), 13)
        self.assertEqual(joker2.get_value(), 13)

    def test_meld_length_property(self):
        """len() returns number of cards."""
        empty = Meld()
        single = Meld(PlayingCard(0))
        double = Meld(PlayingCard(1), single)
        
        self.assertEqual(len(empty), 0)
        self.assertEqual(len(single), 1)
        self.assertEqual(len(double), 2)

    def test_comparing_meld_to_non_meld(self):
        """Comparing meld to non-meld returns NotImplemented."""
        meld = Meld(PlayingCard(0))
        
        # These comparisons should raise TypeError or return NotImplemented
        with self.assertRaises(TypeError):
            _ = meld > 5
        
        with self.assertRaises(TypeError):
            _ = meld == "not a meld"


class TestComplexScenarios(unittest.TestCase):
    """Test complex game scenarios."""

    def test_progression_single_cards(self):
        """Test progression through single cards."""
        cards = [Meld(PlayingCard(i * 4)) for i in range(14)]  # 3 through Joker
        
        for i in range(len(cards) - 1):
            self.assertTrue(cards[i + 1] > cards[i],
                          f"Card {i+1} should beat card {i}")

    def test_cannot_beat_with_wrong_count(self):
        """Cannot beat a double with a triple, etc."""
        double_3 = Meld(PlayingCard(1), Meld(PlayingCard(0)))
        triple_4 = Meld(PlayingCard(10), Meld(PlayingCard(9), Meld(PlayingCard(8))))
        
        # Triple cannot beat double (different lengths)
        self.assertFalse(triple_4 > double_3)

    def test_2_special_rules_comprehensive(self):
        """Comprehensive test of 2's special properties."""
        single_2 = Meld(PlayingCard(48))
        double_2 = Meld(PlayingCard(49), Meld(PlayingCard(48)))
        triple_2 = Meld(PlayingCard(50), Meld(PlayingCard(49), Meld(PlayingCard(48))))
        
        double_ace = Meld(PlayingCard(45), Meld(PlayingCard(44)))
        triple_ace = Meld(PlayingCard(46), Meld(PlayingCard(45), Meld(PlayingCard(44))))
        quad_ace = Meld(PlayingCard(47), Meld(PlayingCard(46), 
                   Meld(PlayingCard(45), Meld(PlayingCard(44)))))
        
        # Single 2 beats double
        self.assertTrue(single_2 > double_ace)
        
        # Double 2 beats triple
        self.assertTrue(double_2 > triple_ace)
        
        # Triple 2 beats quad
        self.assertTrue(triple_2 > quad_ace)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
