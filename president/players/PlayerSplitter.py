#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlayerSplitter extends PlayerSimple with one additional heuristic:
if every non-splitting option would force a pass, split the highest
available card to stay in the round rather than passing.

Strategy:
  - Play the lowest meld that doesn't split a set (same as PlayerSimple)
  - Exception: if passing is the only non-splitting option but other
    plays exist, split the highest card instead to stay in the hand.
  - Always plays 2s and Jokers without hesitation.
"""
import logging
from president.players.PlayerSimple import PlayerSimple


class PlayerSplitter(PlayerSimple):

    def play(self, valid_plays):
        # Scan for the lowest safe (non-splitting) play
        for meld in valid_plays:
            if self._is_safe_to_play(meld):
                if not meld.cards and len(valid_plays) > 1:
                    # Normally, pass, but consider splitting a pair to stay in
                    # split the highest card to stay in the round
                    # TODO: maybe not the _highest_, but median or some heuristic?
                    highest = valid_plays[-2]
                    # Do not split low cards (< 10)
                    if highest.value() > 7:
                        logging.info(
                            f'{self.name}: choosing to split {highest} '
                            f'rather than pass'
                        )
                        return highest
                return meld

        # Should never reach here
        raise RuntimeError(
            f'{self.name}: no play found in {valid_plays}. '
            f'possible_plays() must always include a pass option.'
        )
