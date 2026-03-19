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

    def play(self):
        possible_plays = self.possible_plays()

        # No target — defer to PlayerSimple
        if not self.target_meld:
            return super().play()

        # Scan for the lowest safe (non-splitting) play
        for meld in possible_plays:
            if self._is_safe_to_play(meld):
                if not meld.cards and len(possible_plays) > 1:
                    # Passing is safe but we have other options —
                    # split the highest card to stay in the round
                    highest = possible_plays[-2]
                    logging.info(
                        f'{self.name}: choosing to split {highest} '
                        f'rather than pass'
                    )
                    return highest
                return meld

        # Should never reach here
        raise RuntimeError(
            f'{self.name}: no play found in {possible_plays}. '
            f'possible_plays() must always include a pass option.'
        )