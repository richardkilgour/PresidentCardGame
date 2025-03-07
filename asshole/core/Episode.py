#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Episode module for controlling a single episode of a card game.

This class is responsible for shuffling, dealing, swapping cards, managing rounds,
tracking the highest meld, and finalizing positions.
"""
import logging
import pickle
from enum import Enum
from random import shuffle

from asshole.core.AbstractPlayer import AbstractPlayer
from asshole.core.CardGameListener import CardGameListener
from asshole.core.Meld import Meld
from asshole.core.PlayingCard import PlayingCard


class State(Enum):
    INITIALISED = 1
    DEALING = 2
    SWAPPING = 3
    ROUND_STARTING = 4
    PLAYING = 5
    HAND_WON = 6
    FINISHED = 7


class Episode:
    def __init__(self, players: list[AbstractPlayer], positions: list[AbstractPlayer] | None, deck: list[PlayingCard], listener_list: list[CardGameListener]) -> None:
        """
        Initialize an Episode.

        Args:
            players (list[Any]): list of player objects.
            positions (list[Any]): Starting positions of players; None if a new game.
            deck (list[Any]): list representing the deck of cards.
            listener_list (list[Any]): list of listener objects for notifications.
        """
        self.state: State = State.INITIALISED
        self.players = players
        self.positions = positions
        # These can be inferred from player status
        # '␆' represents "not played yet" for melds; otherwise, melds can be of any type that supports comparison.
        self.current_melds = ['␆', '␆', '␆', '␆']
        self.active_players = []

        self.deck = deck
        self.listener_list = listener_list
        # For testing - only used as a 'checksum' for the cards
        self.discards = []

    def clear(self):
        self.state: State = State.INITIALISED
        self.players = []
        self.positions = []
        self.current_melds = ['␆', '␆', '␆', '␆']
        self.active_players = []
        self.deck = []
        self.listener_list = []
        self.discards = []


    def target_meld(self) -> Meld | None:
        """
        Find the highest meld currently played.

        Returns:
            The highest meld, or None if no valid meld has been played.
        """
        highest_meld = None
        for m in self.current_melds:
            if m and m != '␆' and (highest_meld is None or m > highest_meld):
                highest_meld = m
        return highest_meld

    def swap_cards(self) -> None:
        """
        Perform the card swapping between players based on their positions.
        Assumes positions are set as [King, Citizen, Prince, Asshole].
        """
        if all(item is not None for item in self.positions):
            # Use descriptive names for clarity.
            king, citizen, prince, asshole = self.positions[0], self.positions[1], self.positions[2], self.positions[3]
            tribute = [asshole._hand[-2], asshole._hand[-1]]
            asshole.surrender_cards(tribute, king)
            discard = [king._hand[0], king._hand[1]]
            king.surrender_cards(discard, asshole)
            logging.debug(
                f'{asshole.name} swapped {tribute[0]} and {tribute[1]} for {king.name}\'s cards {discard[0]} and {discard[1]}')
            self.notify_listeners("notify_cards_swapped", king, asshole, 2)

            tribute = [prince._hand[-1]]
            prince.surrender_cards(tribute, citizen)
            discard = [citizen._hand[0]]
            citizen.surrender_cards(discard, prince)
            logging.debug(f'{prince.name} swapped {tribute[0]} for {citizen.name}\'s card {discard[0]}')
            self.notify_listeners("notify_cards_swapped", prince, citizen, 1)

            for player in self.players:
                logging.debug(f'{player.name} has {player}')

    def pick_round_starter(self) -> None:
        """
        Decide and set the starting player for the round.

        If positions exist, the Asshole (positions[3]) starts.
        Otherwise, the player holding the 3♠ (assumed to be value 0, suit 0) starts.
        Notifies listeners about the hand start.
        """
        if all(item is not None for item in self.positions):
            self.move_to_front(self.positions[3])
        else:
            player_with_3_spades = self.find_card_holder(0, 0)
            self.move_to_front(player_with_3_spades)
        self.notify_listeners("notify_hand_start")
        self.notify_listeners("notify_player_turn", self.players[0])

    def notify_listeners(self, notify_func_name: str, *args) -> None:
        """
        Notify all listeners by invoking the specified function with given arguments.

        Args:
            notify_func_name (str): The function name to call on each listener.
            *args: Arguments to pass to the listener function.
        """
        for p in self.listener_list:
            getattr(p, notify_func_name)(*args)

    def move_to_front(self, front_player: AbstractPlayer) -> None:
        """
        Rotate the players list until the specified player is at the front.

        Args:
            front_player (Any): The player to move to the front.
        """
        while self.players[0] != front_player:
            self.players.append(self.players.pop(0))

    def deal(self) -> None:
        """
        Deal the deck of cards to players.

        Ensures that all players start with no cards and that the discard pile is empty.
        Deals cards in a round-robin fashion starting from player index 1.
        """
        for player in self.players:
            assert player.report_remaining_cards() == 0, f"{player.name} has {player.report_remaining_cards()} remaining cards before dealing."
        assert len(self.discards) == 0, "Discards pile is not empty before dealing."

        i = 1
        self.deck.reverse()
        while self.deck:
            self.players[i].card_to_hand(self.deck.pop())
            i = (i + 1) % 4

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

    def players_with_cards(self) -> list[AbstractPlayer]:
        """
        Retrieve a list of players who still have cards in hand.

        Returns:
            list[Any]: Active players with remaining cards.
        """
        active_players = []
        for player in self.players:
            if player.report_remaining_cards() == 0:
                assert player in self.positions, f"{player.name} should be in positions."
            else:
                active_players.append(player)
        return active_players

    def post_episode_checks(self) -> None:
        """
        Perform post-episode checks:
          - Log the final position of the Asshole.
          - Verify the deck and discard pile sizes.
          - Display player rankings.
        """
        asshole = self.positions[-1]
        logging.info(f'{asshole.name} is the Asshole!!! Left with {asshole._hand}')
        while asshole._hand:
            self.discards.append(asshole._hand.pop())
        assert len(self.discards) == 54, "Discards pile should have 54 cards."
        assert len(self.deck) == 0, "Deck should be empty before transferring discards."

        while self.discards:
            self.deck.append(self.discards.pop())

        assert not any(pos is None for pos in self.positions), "There should no empty positions."
        for i, p in enumerate(self.positions):
            logging.info(f"{p.name} is ranked as {p.ranking_names[i]}")
        for p in self.players:
            assert p.report_remaining_cards() == 0, f"{p.name} still has cards remaining."
        assert len(self.deck) == 54, "Deck should have 54 cards after restoration."

    def set_player_finished(self, player: AbstractPlayer) -> None:
        """
        Mark a player as finished and assign them a ranking based on finish order.
        Notifies listeners about the player's finish.

        Args:
            player (Any): The player who has finished playing.
        """
        ranking = self.positions.index(None)
        logging.info(f"{player.name} played out and is ranked {player.ranking_names[ranking]}")
        player.set_position(ranking)
        self.notify_listeners("notify_played_out", player, ranking)
        self.positions[ranking] = player
        player_names = ' '.join(p.name if p else '' for p in self.positions)
        logging.info(f"Positions: {player_names}")

    def player_turn(self) -> None:
        """
        Execute a single turn for the current active player.

        Manages passing, playing a card, updating melds, and player status.
        """
        assert self.active_players, "No active players available for turn."
        logging.info(f"Players who have not passed = {' '.join(x.name for x in self.active_players)}")

        if len(self.active_players) == 1:
            self.set_player_finished(self.active_players[0])
            return

        current_target = self.target_meld()
        logging.info(f'Currently played highest card = {current_target}')

        player = self.active_players[0]
        if player.report_remaining_cards() == 0:
            self.active_players.remove(player)
            return

        self.notify_listeners("notify_player_turn", player)

        action = player.play()
        if action == '␆':  # '␆' indicates a no-op action.
            return

        if current_target and action.cards and action < current_target:
            raise ValueError("Invalid play: action meld is lower than current target meld.")

        self.notify_listeners("notify_play", player, action)
        if not action.cards:
            self.active_players.remove(player)
        else:
            for card in action.cards:
                player._hand.remove(card)
                self.discards.append(card)
            index = self.players.index(player)
            self.current_melds[index] = action
            logging.debug(f'{player.name} is left with {player}')
            if player.report_remaining_cards() == 0:
                self.set_player_finished(player)
            self.active_players.append(self.active_players.pop(0))

    def step(self) -> list[AbstractPlayer]:
        """
        Run the episode state machine until the episode is finished.

        Returns:
            list[Any]: Final rankings/positions of players.
        """
        if self.state == State.INITIALISED:
            shuffle(self.deck)
            self.state = State.DEALING
            self.deal()
        if self.state == State.DEALING:
            # Transition state; potential for delays/notifications.
            self.state = State.SWAPPING
            for player in self.players:
                logging.debug(f'{player.name} has {player}')
            self.swap_cards()
        if self.state == State.SWAPPING:
            self.state = State.ROUND_STARTING
            self.pick_round_starter()
            self.positions = [None, None, None, None]
        if self.state == State.ROUND_STARTING:
            self.active_players = self.players_with_cards()
            self.current_melds = ['␆', '␆', '␆', '␆']
            self.state = State.PLAYING
        if self.state == State.PLAYING:
            self.player_turn()
            if len(self.active_players) == 1:
                self.state = State.HAND_WON
        if self.state == State.HAND_WON:
            if all(item is not None for item in self.positions):
                self.post_episode_checks()
                self.state = State.FINISHED
            else:
                assert len(self.active_players) == 1, "Expected one active player in HAND_WON state."
                self.notify_listeners("notify_hand_won", self.active_players[0])
                self.move_to_front(self.active_players[0])
                self.state = State.ROUND_STARTING

        return self.positions

    def save_state(self) -> bytes:
        """
        Serialize the current game state including each player's hand, meld, state, and positions.

        Returns:
            bytes: Serialized game state.
        """
        game_state = []
        player_names = []
        player_types = []
        for player in self.players:
            game_state.append(player.encode())
            player_names.append(player.name)
            player_types.append(player.__class__.__name__)
        # Mark active player; assumes players[0] is active.
        game_state[0][2, 27] = 1
        serialized = pickle.dumps((player_names, player_types, game_state), protocol=0)
        return serialized

    @staticmethod
    def split_array(array, num_splits: int):
        """
        Split a 2D array into a specified number of splits along the given axis.

        Args:
            array (Any): 2D array-like object.
            num_splits (int): Number of splits.

        Returns:
            list[Any]: list of sub-arrays.

        Raises:
            ValueError: If axis is not 1.
        """
        split_size = len(array[0]) // num_splits
        return [array[:, i * split_size:(i + 1) * split_size] for i in range(num_splits)]

    def restore_state(self, serialized: bytes) -> "Episode":
        """
        Restore the game state from a serialized state.
        WARNING: Uses eval to restore player classes which may be a security risk.

        Args:
            serialized (bytes): Serialized game state.

        Returns:
            Episode: The current episode with restored state.
        """
        deserialized_a = pickle.loads(serialized)
        self.clear()  # Assumes a clear() method exists to reset state.
        player_names = deserialized_a[0]
        player_types = deserialized_a[1]
        game_state = deserialized_a[2]
        for i, name in enumerate(player_names):
            player_class = eval(player_types[i])
            player = self.make_player(player_class, name)
            hand_meld = self.split_array(game_state[1], 2)
            hand = hand_meld[0]
            meld = hand_meld[0]
            player._hand = player.decode_hand(hand)
            if game_state[i][2, 13] == 1:
                player.set_status("passed")
            elif game_state[i][3, 13] == 1:
                player.set_status("waiting")
            else:
                player.set_status(player.decode_hand(meld))
        return self

    def snapshot(self) -> None:
        """
        Create a snapshot of the current game state.
        Note: This function is currently not functional due to issues with state restoration.
        """
        state = self.save_state()
        self.restore_state(state)
