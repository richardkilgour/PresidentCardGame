import logging

from president.core.AbstractPlayer import AbstractPlayer
from president.core.CardGameListener import CardGameListener
from president.core.DeckManager import DeckManager
from president.core.PlayerManager import PlayerManager
from president.core.PlayingCard import PlayingCard
from president.core.Episode import Episode, State


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
        self.player_manager = PlayerManager()
        # List of the current positions of the players (President, etc)
        # Put the winning player at the front of the list. Loser is last
        self.positions = []

        # This is a list of all the card objects - they will be moved around
        self.deck = DeckManager(deck_size)

        self.listener_list = []
        self.number_of_rounds = None
        self.round_number = 0
        # Episode contains current player hands and status (playing, meld, out or passed)
        self.episode = None

    def clear(self) -> None:
        """Reset the GameMaster by clearing all players, positions, and listeners."""
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

    def get_player_status_from_episode(self, player):
        if player in self.episode.active_players:
            player_index = self.player_manager.players.index(player)
            # If they have played card(s), return that
            if self.episode.current_melds[player_index] and self.episode.current_melds[player_index] != '␆':
                return self.episode.current_melds[player_index]
            else:
                return "Waiting"
        else:
            # No position and not active -> Passed
            return "Passed"

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
            return self.get_player_status_from_episode(player)
        elif player in self.player_manager.players:
            return "Waiting"
        else:
            return "Absent"

    def notify_player_joined(self, player: AbstractPlayer, position:int):
        # Notify everyone of the new player
        self.notify_listeners("notify_player_joined", player, position)
        # Notify the new player of the other players (include own position)
        for i, existing_player in enumerate(self.player_manager.players):
            if existing_player:
                player.notify_player_joined(existing_player, i)
        self.add_listener(player)


    def add_player(self, player: AbstractPlayer, position:int|None = None) -> int:
        position = self.player_manager.add_player(player, position)
        self.notify_player_joined(player, position)
        return position


    def make_player(self, player_type: type[AbstractPlayer], name: str = None) -> AbstractPlayer | None:
        # Forward this to the Player manager
        new_player =  self.player_manager.make_player(player_type, name)
        self.add_player(new_player)
        return new_player

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
            self.deck.shuffle()
        self.episode = Episode(self.player_manager, [], self.deck, self.listener_list)

    def start(self, number_of_rounds: int = 100, preset_hands: list[PlayingCard] = None) -> None:
        """
        Start playing a series of game rounds.

        Args:
            number_of_rounds: Number of rounds to play. Set to None for infinite play.
            preset_hands: Optional pre-arranged deck for testing or tournament play

        Raises:
            Exception: If there are not exactly 4 players
        """
        if None in self.player_manager.players:
            raise Exception("Not enough Players")
        self.round_number = 0
        self.number_of_rounds = number_of_rounds
        # The initial hand has no positions
        self.reset(preset_hands=preset_hands)
        self.notify_listeners("notify_game_stated")

    def setup_new_round(self):
        self.positions = self.episode.ranks
        logging.info(f"--- Round Finished with positions {self.positions} ---")
        self.reset()
        # Keep some stats
        self.round_number += 1
        if self.number_of_rounds and self.round_number > self.number_of_rounds:
            print(self.position_stats_str())
            self.remove_worst_player()
            return True
        return False

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
            return self.setup_new_round()
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
        for player in self.player_manager.players:
            result.append(
                f'{player.name} was President {player.position_count[0]}; '
                f'Vice-President {player.position_count[1]}; '
                f'Citizen {player.position_count[2]} and '
                f'Scumbag {player.position_count[3]}. '
                f'Score = {player.get_score()}'
            )
        return '\n'.join(result)


    def remove_worst_player(self) -> None:
        """
        Remove the player with the lowest score from the game.
        """
        worst_player = None
        lowest_score = float('inf')  # Start with infinity to find minimum

        for player in self.player_manager.players:
            score = player.get_score()
            if score < lowest_score:
                worst_player = player
                lowest_score = score

        if worst_player:
            print(f'{worst_player.name} is pissed off and quits')
            self.player_manager.players.remove(worst_player)
