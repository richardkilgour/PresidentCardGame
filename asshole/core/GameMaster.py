import logging
from random import shuffle

from asshole.core.AbstractPlayer import AbstractPlayer
from asshole.core.CardGameListener import CardGameListener
from asshole.core.PlayingCard import PlayingCard
from asshole.core.Episode import Episode, State


class GameMaster:
    """
    A class to control all the action in the card game.

    Manages players, card distribution, game flow, and event notifications.
    """

    def __init__(self, deck_size: int = 54) -> None:
        """
        Initialize a new GameMaster.

        Args:
            deck_size: Number of cards in the deck. Defaults to 54.
        """
        self.players = [None, None, None, None]
        self.positions = [None, None, None, None]
        # This is a list of all the card objects - they will be moved around
        self.deck_size: int = deck_size
        self.deck = [PlayingCard(i) for i in range(deck_size)]
        self.listener_list = []
        self.number_of_rounds = None
        self.round_number = 0
        # Episode contains current player hands and status (playing, meld, out or passed)
        self.episode = None

    def clear(self) -> None:
        """Reset the GameMaster by clearing all players, positions, and listeners."""
        self.players = [None, None, None, None]
        self.positions = [None, None, None, None]
        self.listener_list = []

    def add_listener(self, listener: CardGameListener) -> None:
        """
        Add a listener to be notified of game events.

        Args:
            listener: The CardGameListener to add
        """
        self.listener_list.append(listener)

    def notify_listeners(self, notify_func_name: str, *args) -> None:
        """
        Notify all listeners of a game event.

        Args:
            notify_func_name: The name of the listener method to call
            *args: Arguments to pass to the listener method
        """
        for listener in self.listener_list:
            getattr(listener, notify_func_name)(*args)

    def get_player_status(self, player: AbstractPlayer) -> str:
        """
        Get the status of a player.

        Args:
            player: The player to get the status of

        Returns:
            A string indicating the player's status:
            - "Absent"
            - "Waiting" (not yet played)
            - "Passed"
            - "Played" (with the cards played)
        """
        # Query the Episode and get the current status of the requested player
        if self.episode:
            if player in self.episode.active_players:
                player_index = self.players.index(player)
                # If they have played card(s), return that
                if self.episode.current_melds[player_index] and self.episode.current_melds[player_index] != '␆':
                    return self.episode.current_melds[player_index]
                else:
                    return "Waiting"
            else:
                # No position and not active -> Passed
                return "Passed"
        elif player in self.players:
            return "Waiting"
        else:
            return "Absent"

    def add_player(self, player: AbstractPlayer, position:int = None) -> int:
        """
        Add a player to the game.

        Args:
            player: The player to add
            name: Optional name for the player
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

        # Notify everyone of the new player
        self.notify_listeners("notify_player_joined", player, position)
        # Notify the new player of the other players (include own position)
        self.players[position] = player
        for i, existing_player in enumerate(self.players):
            if existing_player:
                player.notify_player_joined(existing_player, i)
        self.add_listener(player)
        return position

    def make_player(self, player_type: type[AbstractPlayer], name: str = None) -> AbstractPlayer:
        """
        Create a new player and add them to the game.

        Args:
            player_type: Either a class or a string (for a saved/serialized player)
            name: Optional name for the player

        Returns:
            The newly created player
        """
        if not name:
            name = f'Player {len(self.players)}'
        new_player = player_type(name)
        new_player_index = self.add_player(new_player)
        # Return the new player
        return self.players[new_player_index]

    def reset(self, preset_hands: list[PlayingCard] = None) -> None:
        """
        Reset the game state with a fresh deck.

        Args:
            preset_hands: Optional pre-arranged deck for testing or tournament play
        """
        logging.info(f"--- Start of a new Game --- {self.positions=}")
        # A client can shuffle the cards themselves (for testing... or cheating?)
        # Shuffle the cards
        if not preset_hands:
            shuffle(self.deck)
        self.episode = Episode(self.players, self.positions, self.deck, self.listener_list)

    def start(self, number_of_rounds: int = 100, preset_hands: list[PlayingCard] = None) -> None:
        """
        Start playing a series of game rounds.

        Args:
            number_of_rounds: Number of rounds to play. Set to None for infinite play.
            preset_hands: Optional pre-arranged deck for testing or tournament play

        Raises:
            Exception: If there are not exactly 4 players
        """
        if None in self.players:
            raise Exception("Not enough Players")
        self.round_number = 0
        self.number_of_rounds = number_of_rounds
        # The initial hand has no positions
        self.reset(preset_hands=preset_hands)
        self.notify_listeners("notify_game_stated")

    def step(self) -> bool:
        """
        Execute a single step of the game.

        Returns:
            True if all rounds are finished, False otherwise
        """
        if not hasattr(self, 'episode') or self.episode is None:
            raise RuntimeError("Game has not been started. Call start() first.")

        if self.episode.state == State.FINISHED:
            # Set up a new round
            self.positions = self.episode.positions
            logging.info(f"--- Round Finished with positions {self.positions} ---")
            self.reset()
            assert len(self.deck) == 54, "Deck size should be 54 after reset"
            # Keep some stats
            self.round_number += 1
            if self.number_of_rounds and self.round_number > self.number_of_rounds:
                print(self.position_stats_str())
                self.remove_worst_player()
                return True
        else:
            self.episode.step()
        return False

    def position_stats_str(self) -> str:
        """
        Return a string with statistics about each player's positions.

        Returns:
            A formatted string with position statistics for each player
        """
        result = []
        for player in self.players:
            result.append(
                f'{player.name} was King {player.position_count[0]}; '
                f'Prince {player.position_count[1]}; '
                f'Citizen {player.position_count[2]} and '
                f'Asshole {player.position_count[3]}. '
                f'Score = {player.get_score()}'
            )
        return '\n'.join(result)


    def remove_worst_player(self) -> None:
        """
        Remove the player with the lowest score from the game.
        """
        worst_player = None
        lowest_score = float('inf')  # Start with infinity to find minimum

        for player in self.players:
            score = player.get_score()
            if score < lowest_score:
                worst_player = player
                lowest_score = score

        if worst_player:
            print(f'{worst_player.name} is pissed off and quits')
            self.players.remove(worst_player)