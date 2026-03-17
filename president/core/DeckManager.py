"""
DeckManager.py

A class to manage the deck of cards for the President card game.

This class is responsible for:
- Creating and shuffling the deck
- Dealing cards to players
- Managing the remaining deck and discarded cards

Dependencies:
    - president.core.PlayingCard: Represents a playing card.
"""
from random import shuffle
from typing import List

from president.core.AbstractPlayer import AbstractPlayer
from president.core.PlayingCard import PlayingCard

class DeckManager:
    """
    A class to manage the deck of cards for the game.

    Attributes:
        deck_size (int): Number of cards in the deck.
        deck (List[PlayingCard]): List of PlayingCard objects representing the deck.
        discarded_cards (List[PlayingCard]): List of discarded cards.
    """

    def __init__(self, deck_size: int = 54) -> None:
        """
        Initialize a new DeckManager.

        Args:
            deck_size: Number of cards in the deck. Defaults to 54.
        """
        self._deck_size = deck_size
        self.deck = self.reset_deck()
        self.shuffle()

    def size(self):
        return self._deck_size

    def shuffle(self) -> None:
        shuffle(self.deck)

    def give_next_card_to_player(self, player):
        player.card_to_hand(self.deck.pop())

    def deal_cards(self, players : list[AbstractPlayer]):
        i = 1
        self.deck.reverse()
        while self.deck:
            self.give_next_card_to_player(players[i])
            i = (i + 1) % len(players)
        assert len(self.deck) == 0

    def reset_deck(self) -> list[PlayingCard]:
        return [PlayingCard(i) for i in range(self._deck_size)]

