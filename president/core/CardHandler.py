#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CardHandler manages the physical movement of cards between zones:
  - Player hands
  - The discard pile
  - The deck

Responsibilities:
  - Swap cards between players at the start of a new episode
  - Move played cards from a player's hand to the discard pile
  - Restore the deck from the discard pile at the end of an episode
  - Clear the scumbag's remaining cards into the discard pile

Not responsible for: game rules, turn flow, or listener notification.
Notification is handled by Episode, which calls CardHandler methods
and then notifies listeners itself.
"""
import logging
from typing import Optional, Any

from president.core.AbstractPlayer import AbstractPlayer
from president.core.DeckManager import DeckManager
from president.core.Meld import Meld
from president.core.PlayingCard import PlayingCard


class CardHandler:
    def __init__(self, deck: DeckManager) -> None:
        """
        Args:
            deck: The DeckManager owning the physical cards.
            discarded_cards: Shared reference to PlayerManager.discarded_cards.
        """
        self.deck = deck
        self.discard_pile: list[PlayingCard] = []

    # -------------------------------------------------------------------------
    # Pre-episode card swap
    # -------------------------------------------------------------------------

    def swap_for_new_episode(self, ranks: list[AbstractPlayer]) -> Optional[tuple[tuple[Any, Any], tuple[Any, Any]]]:
        """
        Swap cards between players based on their rankings from the previous episode.
        President receives 2 best cards from Scumbag.
        Vice-President receives 1 best card from Citizen.

        Args:
            ranks: [president, vice_president, citizen, scumbag]
        """
        if len(ranks) != 4:
            return
        president, vice_president, citizen, scumbag = ranks
        best_2, worst_2 = self._swap(scumbag, president, 2)
        best_1, worst_1 = self._swap(citizen, vice_president, 1)
        for player in [president, vice_president, citizen, scumbag]:
            logging.debug(f'{player.name} now has {player}')
        return (best_2, worst_2), (best_1, worst_1)

    def _swap(self, low_player: AbstractPlayer, high_player: AbstractPlayer, num_cards: int):
        """
        Swap num_cards between two players.
        Low player gives their best cards; high player gives their worst.

        Returns:
            Tuple of (cards given by low_player, cards given by high_player)
        """
        best_n = low_player._hand[-num_cards:]
        low_player.surrender_cards(best_n, high_player)

        worst_n = high_player._hand[:num_cards]
        high_player.surrender_cards(worst_n, low_player)

        logging.debug(
            f'{low_player.name} swapped {", ".join(map(str, best_n))} '
            f'for {high_player.name}\'s {", ".join(map(str, worst_n))}'
        )
        return best_n, worst_n

    # -------------------------------------------------------------------------
    # During play
    # -------------------------------------------------------------------------

    def play_meld(self, player: AbstractPlayer, meld: Meld) -> None:
        """
        Move the cards in a meld from the player's hand to the discard pile.

        Args:
            player: The player who played the meld.
            meld: The meld being played.
        """
        for card in meld.cards:
            for i, h in enumerate(player._hand):
                if h.same_card(card):
                    self.discard_pile.append(player._hand.pop(i))
                    break
        logging.debug(f'{player.name} is left with {player}')

    # -------------------------------------------------------------------------
    # Post-episode cleanup
    # -------------------------------------------------------------------------

    def collect_scumbag_cards(self, scumbag: AbstractPlayer) -> None:
        """
        Move the scumbag's remaining cards to the discard pile.

        Args:
            scumbag: The last-placed player who still holds cards.
        """
        logging.info(f'{scumbag.name} is the Scumbag!!! Left with {" ".join(str(c) for c in scumbag._hand)}')
        while scumbag._hand:
            self.discard_pile.append(scumbag._hand.pop())

    def restore_deck(self) -> None:
        """
        Move all discarded cards back into the deck, ready for the next episode.
        Asserts correct card counts before and after.
        """
        assert len(self.discard_pile) == 54, \
            f"Expected 54 discarded cards before restore, got {len(self.discard_pile)}."
        assert len(self.deck.deck) == 0, \
            "Deck should be empty before restoring from discards."
        while self.discard_pile:
            self.deck.deck.append(self.discard_pile.pop())
        assert len(self.deck.deck) == 54, \
            "Deck should have 54 cards after restoration."