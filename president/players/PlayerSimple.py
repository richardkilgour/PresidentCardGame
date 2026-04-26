#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A simple AI player that plays the lowest possible meld without splitting a set.

Strategy:
  - Always play the lowest meld that beats the current target.
  - Never split a set (e.g. won't play one 7 if holding two 7s).
  - Exception: 2s and Jokers are always played even if it splits them.
  - If every available play would split a set, pass instead.
"""
import logging
from president.core.AbstractPlayer import AbstractPlayer


class PlayerSimple(AbstractPlayer):
    """Plays the lowest possible meld, avoiding splits."""

    def play(self, valid_plays):
        for meld in valid_plays:
            if self._is_safe_to_play(meld):
                return meld
            logging.info(
                f'{self.name}: skipping {meld} to avoid splitting '
                f'{self.number_of_cards_of_value(meld.cards[0].get_value())}x '
                f'{meld.cards[0]}'
            )

    def _is_safe_to_play(self, meld) -> bool:
        """
        Returns True if playing this meld will not split a set,
        or if the meld is a pass (always safe to take if no better option exists).
        2s and Jokers are always safe to play regardless of splits.
        """
        if not meld.cards:
            return True  # Pass is always acceptable as a last resort
        card_value = meld.cards[0].get_value()
        is_high_card = card_value > 11  # 2s (12) and Jokers (13)
        plays_full_set = self.number_of_cards_of_value(card_value) == len(meld)
        return is_high_card or plays_full_set