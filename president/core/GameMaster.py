import logging

from president.core.AbstractPlayer import AbstractPlayer
from president.core.CardGameListener import CardGameListener
from president.core.DeckManager import DeckManager
from president.core.PlayerManager import PlayerManager
from president.core.PlayingCard import PlayingCard
from president.core.Episode import Episode, State


class GameMaster:
    """
    Controls the overall flow of the card game across multiple rounds.

    Responsibilities: player registration, round lifecycle, listener
    management, and end-of-tournament statistics.

    Not responsible for: turn-by-turn game logic (Episode),
    card movement mechanics (CardHandler - future).
    """

    def __init__(self, deck_size: int = 54) -> None:
        self.player_manager = PlayerManager()
        self.positions = []  # Rankings from the last completed episode
        self.deck = DeckManager(deck_size)
        self.listener_list = []
        self.number_of_rounds = None
        self.round_number = 0
        self.episode = None

    def clear(self) -> None:
        """Reset listeners. Does not remove players."""
        self.listener_list = []

    # -------------------------------------------------------------------------
    # Listener management
    # -------------------------------------------------------------------------

    def add_listener(self, listener: CardGameListener) -> None:
        self.listener_list.append(listener)

    def notify_listeners(self, notify_func_name: str, *args) -> None:
        for listener in self.listener_list:
            getattr(listener, notify_func_name)(*args)

    # -------------------------------------------------------------------------
    # Player management
    # -------------------------------------------------------------------------

    def notify_player_joined(self, player: AbstractPlayer, position: int) -> None:
        """Notify all existing listeners of the new player, then add the player as a listener."""
        self.notify_listeners("notify_player_joined", player, position)
        # Catch the new player up on who is already seated
        for i, existing_player in enumerate(self.player_manager.players):
            if existing_player:
                player.notify_player_joined(existing_player, i)
        self.add_listener(player)

    def add_player(self, player: AbstractPlayer, position: int | None = None) -> int:
        position = self.player_manager.add_player(player, position)
        self.notify_player_joined(player, position)
        return position

    def make_player(self, player_type: type[AbstractPlayer], name: str = None) -> AbstractPlayer | None:
        new_player = self.player_manager.make_player(player_type, name)
        self.add_player(new_player)
        return new_player

    # -------------------------------------------------------------------------
    # Game lifecycle
    # -------------------------------------------------------------------------

    def reset(self, preset_hands: list[PlayingCard] = None) -> None:
        """
        Create a new Episode, passing through the rankings from the previous one
        so card swapping and turn order are set correctly.
        """
        logging.info(f"--- Start of a new Episode --- {self.positions=}")
        if not preset_hands:
            self.deck.shuffle()
        self.episode = Episode(self.player_manager, self.positions, self.deck, self.listener_list)

    def start(self, number_of_rounds: int = 100, preset_hands: list[PlayingCard] = None) -> None:
        """
        Start a series of episodes.

        Args:
            number_of_rounds: Number of episodes to play. None for infinite.

        Raises:
            Exception: If there are fewer than 4 players.
        """
        if None in self.player_manager.players:
            raise Exception("Not enough Players")
        self.round_number = 0
        self.number_of_rounds = number_of_rounds
        self.positions = []  # No rankings for the first episode
        self.reset(preset_hands=preset_hands)
        self.notify_listeners("notify_game_stated")

    def setup_new_round(self) -> bool:
        """
        Finalise the current episode and set up the next one.

        Returns:
            True if the tournament is over, False otherwise.
        """
        self.positions = self.episode.ranks
        logging.info(f"--- Episode Finished with positions {self.positions} ---")
        self.round_number += 1
        if self.number_of_rounds and self.round_number > self.number_of_rounds:
            print(self.position_stats_str())
            self.remove_worst_player()
            return True
        self.reset()
        return False

    def step(self) -> bool:
        """
        Advance the game by one step.

        Returns:
            True if the tournament is finished, False otherwise.
        """
        if self.episode is None:
            raise RuntimeError("Game has not been started. Call start() first.")

        if self.episode.state == State.FINISHED:
            return self.setup_new_round()

        self.episode.step()
        return False

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def position_stats_str(self) -> str:
        """Return a formatted summary of each player's position counts and score."""
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
        """Remove the player with the lowest cumulative score."""
        worst_player = min(self.player_manager.players, key=lambda p: p.get_score())
        if worst_player:
            print(f'{worst_player.name} is pissed off and quits')
            self.player_manager.players.remove(worst_player)