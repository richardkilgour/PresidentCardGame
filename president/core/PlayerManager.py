"""
PlayerManager.py

A class to manage player hands, played/discarded cards, and card swaps.

Responsibilities:
- Track players and update them
- Handle card swaps between players
- Track and update player states
- Keep the original list of players (self.players) It is the seat positions, and never changes.
- Track the next player dynamically
"""
from president.core.PlayingCard import PlayingCard
from president.core.AbstractPlayer import AbstractPlayer

class PlayerManager:
    def __init__(self) -> None:
        """
        Initialize a PlayerManager with no players.
        Class variables are initialized once 4 players are added.
        """
        self.players: list[AbstractPlayer|None] = [None, None, None, None]
        self.hands = {player: [] for player in self.players}
        self.played_cards = []
        self.discarded_cards = []
        self.player_states = {player: "active" for player in self.players}

    def add_player(self, player: AbstractPlayer, position:int|None = None) -> int:
        """
        Add a player to the game.

        Args:
            player: The player to add
            position: Optional position to add the new player there. Otherwise, take the first available one
        Returns:
            The index of the new player
        Raises:
            Exception: ValueError If there are already 4 players
            Exception: If a requested position is taken
        """
        if not position:
            position = self.players.index(None)
        if position and self.players[position]:
            print(f"WARNING: Can't replace exiting player {position=}")
            return -1

        self.players[position] = player

        return position



    def add_to_hand(self, player: AbstractPlayer, cards: list[PlayingCard]) -> None:
        """
        Add cards to a player's hand.

        Args:
            player: The player to receive the cards.
            cards: List of cards to add.
        """
        if player in self.hands:
            self.hands[player].extend(cards)

    def remove_from_hand(self, player: AbstractPlayer, cards: list[PlayingCard]) -> None:
        """
        Remove cards from a player's hand.

        Args:
            player: The player whose cards are being removed.
            cards: List of cards to remove.
        """
        if player in self.hands:
            for card in cards:
                self.hands[player].remove(card)

    def play_cards(self, player: AbstractPlayer, cards: list[PlayingCard]) -> None:
        """
        Play cards from a player's hand.

        Args:
            player: The player playing the cards.
            cards: List of cards to play.
        """
        if self.played_cards is not None and player in self.hands:
            self.remove_from_hand(player, cards)
            self.played_cards.extend(cards)

    def discard_cards(self, cards: list[PlayingCard]) -> None:
        """
        Discard cards.

        Args:
            cards: List of cards to discard.
        """
        if self.discarded_cards is not None:
            self.discarded_cards.extend(cards)

    def swap_cards(self, player1: AbstractPlayer, player2: AbstractPlayer, num_cards: int) -> None:
        """
        Swap cards between two players.

        Args:
            player1: First player.
            player2: Second player.
            num_cards: Number of cards to swap.
        """
        # Implementation: Swap the worst `num_cards` from player1 with the best `num_cards` from player2
        pass

    def has_cards(self, player: AbstractPlayer) -> bool:
        """
        Check if a player has cards.

        Args:
            player: The player to check.

        Returns:
            True if the player has cards, False otherwise.
        """
        return len(self.hands.get(player, [])) > 0

    def clear_played_cards(self) -> None:
        """Clear the list of played cards for a new round."""
        if self.played_cards is not None:
            self.played_cards = []

    def clear_discarded_cards(self) -> None:
        """Clear the list of discarded cards for a new round."""
        if self.discarded_cards is not None:
            self.discarded_cards = []

    def set_player_state(self, player: AbstractPlayer, state: str) -> None:
        """
        Set the state of a player.

        Args:
            player: The player whose state is being updated.
            state: The new state (e.g., "active", "passed", "finished").
        """
        if self.player_states is not None and player in self.player_states:
            self.player_states[player] = state

    def get_player_state(self, player: AbstractPlayer) -> str|None:
        """
        Get the state of a player.

        Args:
            player: The player whose state is being queried.

        Returns:
            The player's current state, or None if not found.
        """
        if self.player_states is not None:
            return self.player_states.get(player)
        return None

    def find_card_holder(self, target_card_value: int, target_card_suit: int) -> AbstractPlayer | None:
        """
        Find the player holding a card with the specified value and suit.
        Assumes cards in a player's hand are unsorted with respect to suit.

        Args:
            target_card_value (int): The target card value.
            target_card_suit (int): The target card suit.

        Returns:
            The player holding the card, or None if not found.
        """
        for player in self.players:
            for card in player._hand:
                if card.get_value() > target_card_value:
                    break
                if card.get_value() == target_card_value and card.get_suit() == target_card_suit:
                    return player
        return None

    def make_player(self, player_type: type[AbstractPlayer], name: str = None) -> AbstractPlayer | None:
        # Forward this to the Player manager
        """
        Create a new player and add them to the game in the first available space

        Args:
            player_type: Either a class or a string (for a saved/serialized player)
            name: Optional name for the player

        Returns:
            The newly created player
        """
        if not name:
            name = f'Player {len(self.players)}'
        new_player = player_type(name)
        # Return the new player
        return new_player
