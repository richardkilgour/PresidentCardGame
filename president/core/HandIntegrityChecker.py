#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HandIntegrityChecker cross-validates every player's play history against
their recorded starting hand at the end of an episode.
"""
import logging

from president.core.PlayerManager import PlayerManager
from president.core.AbstractPlayer import AbstractPlayer

logger = logging.getLogger(__name__)


class HandIntegrityChecker:
    @staticmethod
    def verify(player_manager: PlayerManager, ranks: list[AbstractPlayer]) -> None:
        """
        Cross-check every player's play history against their recorded starting hand.
        Each player's PlayHistory is checked independently as a cross-validation.

        Args:
            player_manager: Provides the full player list.
            ranks:          Finishing order for this episode; ranks[-1] is the scumbag.

        Raises:
            AssertionError: If any reconstructed hand does not match the starting hand.
        """
        scumbag_remaining = list(ranks[-1]._hand)
        for player in player_manager.players:
            memory = player.memory
            existing = scumbag_remaining if player is ranks[-1] else []
            reconstructed = memory.reconstruct_hand(player, existing)
            expected = player._starting_hand
            if [c.get_index() for c in reconstructed] != [c.get_index() for c in expected]:
                raise AssertionError(
                    f"Hand integrity failed for {player.name} "
                    f"reconstructed {[str(c) for c in reconstructed]} "
                    f"!= starting hand {[str(c) for c in expected]}"
                )
        logger.info("Hand integrity check passed for all players.")
