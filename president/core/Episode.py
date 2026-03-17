#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Episode module for controlling a single episode of a card game.

Takes players in their current positions and plays through to new positions.
Responsible for: state machine flow, turn management, rank assignment,
and notifying listeners of all game events.

Not responsible for: card movement mechanics (see CardHandler),
serialisation (see GameCheckpoint - future).
"""
import logging
import pickle
from enum import Enum

from president.core.AbstractPlayer import AbstractPlayer
from president.core.CardGameListener import CardGameListener
from president.core.DeckManager import DeckManager
from president.core.Meld import Meld
from president.core.PlayerManager import PlayerManager


class State(Enum):
    INITIALISED = 1
    DEALING = 2
    SWAPPING = 3
    ROUND_STARTING = 4
    PLAYING = 5
    HAND_WON = 6
    FINISHED = 7


class Episode:
    def __init__(self, player_manager: PlayerManager, ranks: list[AbstractPlayer], deck: DeckManager,
                 listener_list: list[CardGameListener]) -> None:
        self.state: State = State.INITIALISED
        self.player_manager = player_manager
        self.ranks = ranks
        # '␆' represents "not played yet"; external dependency, refactor separately
        self.current_melds = ['␆', '␆', '␆', '␆']
        self.active_players = []
        self.deck = deck
        self.listener_list = listener_list

    def clear(self):
        self.state: State = State.INITIALISED
        self.player_manager = None
        self.current_melds = ['␆', '␆', '␆', '␆']
        self.active_players = []
        self.deck = []
        self.listener_list = []

    def notify_listeners(self, notify_func_name: str, *args) -> None:
        for p in self.listener_list:
            getattr(p, notify_func_name)(*args)

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def deal(self) -> None:
        """Deal cards to all players. Asserts clean state before dealing."""
        for player in self.player_manager.players:
            assert player.report_remaining_cards() == 0, \
                f"{player.name} has {player.report_remaining_cards()} remaining cards before dealing."
        assert len(self.player_manager.discarded_cards) == 0, \
            "Discards pile is not empty before dealing."
        self.deck.deal_cards(self.player_manager.players)

    def swap_player_cards(self, low_player, high_player, num_cards):
        # TODO: Move to CardHandler
        best_n_cards = low_player._hand[-num_cards:]
        low_player.surrender_cards(best_n_cards, high_player)
        worst_n_cards = high_player._hand[:num_cards]
        high_player.surrender_cards(worst_n_cards, low_player)
        logging.debug(
            f'{low_player.name} swapped {", ".join(map(str, best_n_cards))} '
            f'for {high_player.name}\'s cards {", ".join(map(str, worst_n_cards))}'
        )
        self.notify_listeners("notify_cards_swapped", high_player, low_player, num_cards)

    def swap_cards(self) -> None:
        """Swap cards between players based on their previous positions."""
        # TODO: Move to CardHandler
        if len(self.ranks) == 4:
            president, vice_president, citizen, scumbag = self.ranks
            self.swap_player_cards(scumbag, president, 2)
            self.swap_player_cards(citizen, vice_president, 1)
            for player in self.player_manager.players:
                logging.debug(f'{player.name} has {player}')

    def move_to_front(self, front_player: AbstractPlayer) -> None:
        """Rotate the players list until the specified player is at the front."""
        while self.player_manager.players[0] != front_player:
            self.player_manager.players.append(self.player_manager.players.pop(0))

    def pick_round_starter(self) -> None:
        """Set the starting player and notify listeners that a new hand is beginning."""
        if len(self.ranks) == 4:
            starting_player = self.ranks[3]  # Scumbag starts
        else:
            starting_player = self.player_manager.find_card_holder(0, 0)  # Holder of 3♠
        self.move_to_front(starting_player)
        self.notify_listeners("notify_hand_start")
        self.notify_listeners("notify_player_turn", self.player_manager.players[0])

    # -------------------------------------------------------------------------
    # Turn management
    # -------------------------------------------------------------------------

    def target_meld(self) -> Meld | None:
        """Return the highest meld currently on the table, or None if none played yet."""
        highest_meld = None
        for m in self.current_melds:
            if m and m != '␆' and (highest_meld is None or m > highest_meld):
                highest_meld = m
        return highest_meld

    def get_players_with_cards(self) -> list[AbstractPlayer]:
        """Return all players who still have cards in hand."""
        active_players = []
        for player in self.player_manager.players:
            if player.report_remaining_cards() == 0:
                assert player in self.ranks, f"{player.name} should have a rank."
            else:
                active_players.append(player)
        return active_players

    def set_player_finished(self, player: AbstractPlayer) -> None:
        """Assign a rank to a player who has played out and notify listeners."""
        ranking = len(self.ranks)
        self.ranks.append(player)
        logging.info(f"{player.name} played out and is ranked {player.ranking_names[ranking]}")
        player.set_position(ranking)
        self.notify_listeners("notify_played_out", player, ranking)
        player_names = ' '.join(p.name if p else '' for p in self.ranks)
        logging.info(f"Positions: {player_names}")

    def player_turn(self) -> None:
        """Execute a single turn for the current active player."""
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
        if action == '␆':  # no-op, async player not ready yet
            return

        if current_target and action.cards and action < current_target:
            raise ValueError("Invalid play: action meld is lower than current target meld.")

        if not action.cards:
            # Player passed
            self.notify_listeners("notify_pass", player)
            self.active_players.remove(player)
        else:
            self.notify_listeners("notify_play", player, action)
            # TODO: Move card movement to CardHandler
            for card in action.cards:
                player._hand.remove(card)
                self.player_manager.discarded_cards.append(card)
            index = self.player_manager.players.index(player)
            self.current_melds[index] = action
            logging.debug(f'{player.name} is left with {player}')
            if player.report_remaining_cards() == 0:
                self.set_player_finished(player)
            self.active_players.append(self.active_players.pop(0))

    # -------------------------------------------------------------------------
    # Post-episode cleanup
    # -------------------------------------------------------------------------

    def post_episode_checks(self) -> None:
        """
        Finalise the episode: log rankings, move scumbag's cards to discard,
        and restore the full deck.
        TODO: Card restoration to move to CardHandler or DeckManager.
        """
        assert len(self.ranks) == 4

        scumbag = self.ranks[-1]
        logging.info(f'{scumbag.name} is the Scumbag!!! Left with {scumbag._hand}')
        while scumbag._hand:
            self.player_manager.discarded_cards.append(scumbag._hand.pop())
        assert len(self.player_manager.discarded_cards) == 54, \
            "Discards pile should have 54 cards."
        assert len(self.deck.deck) == 0, \
            "Deck should be empty before transferring discards."

        while self.player_manager.discarded_cards:
            self.deck.deck.append(self.player_manager.discarded_cards.pop())

        assert not any(pos is None for pos in self.ranks), \
            "There should be no empty positions."
        for i, p in enumerate(self.ranks):
            logging.info(f"{p.name} is ranked as {p.ranking_names[i]}")
        for p in self.player_manager.players:
            assert p.report_remaining_cards() == 0, \
                f"{p.name} still has cards remaining."
        assert len(self.deck.deck) == 54, \
            "Deck should have 54 cards after restoration."

    # -------------------------------------------------------------------------
    # State machine
    # -------------------------------------------------------------------------

    def step(self) -> list[AbstractPlayer | None]:
        """
        Advance the episode state machine by one step.

        Returns:
            Current rankings. Empty until episode is finished.
        """
        if self.state == State.INITIALISED:
            self.deck.shuffle()
            self.state = State.DEALING
            self.deal()
        if self.state == State.DEALING:
            self.state = State.SWAPPING
            for player in self.player_manager.players:
                logging.debug(f'{player.name} has {player}')
            self.swap_cards()
        if self.state == State.SWAPPING:
            self.state = State.ROUND_STARTING
            self.pick_round_starter()
            self.ranks = []
        if self.state == State.ROUND_STARTING:
            self.active_players = self.get_players_with_cards()
            self.current_melds = ['␆', '␆', '␆', '␆']
            # Notify waiting players (those already ranked from a previous hand)
            for player in self.player_manager.players:
                if player not in self.active_players:
                    self.notify_listeners("notify_waiting", player)
            self.state = State.PLAYING
        if self.state == State.PLAYING:
            self.player_turn()
            if len(self.active_players) == 1:
                self.state = State.HAND_WON
        if self.state == State.HAND_WON:
            if len(self.ranks) == 4:
                self.post_episode_checks()
                self.state = State.FINISHED
            else:
                assert len(self.active_players) == 1, \
                    "Expected one active player in HAND_WON state."
                self.notify_listeners("notify_hand_won", self.active_players[0])
                self.move_to_front(self.active_players[0])
                self.state = State.ROUND_STARTING

        return self.ranks

    # -------------------------------------------------------------------------
    # Serialisation — pending extraction to GameCheckpoint
    # -------------------------------------------------------------------------

    def save_state(self) -> bytes:
        """TODO: Move to GameCheckpoint."""
        game_state = []
        player_names = []
        player_types = []
        for player in self.player_manager.players:
            game_state.append(player.encode())
            player_names.append(player.name)
            player_types.append(player.__class__.__name__)
        game_state[0][2, 27] = 1
        return pickle.dumps((player_names, player_types, game_state), protocol=0)

    @staticmethod
    def split_array(array, num_splits: int):
        """TODO: Move to GameCheckpoint."""
        split_size = len(array[0]) // num_splits
        return [array[:, i * split_size:(i + 1) * split_size] for i in range(num_splits)]

    def restore_state(self, serialized: bytes) -> "Episode":
        """TODO: Move to GameCheckpoint. WARNING: uses eval — security risk."""
        deserialized_a = pickle.loads(serialized)
        self.clear()
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
        """TODO: Move to GameCheckpoint. Currently non-functional."""
        state = self.save_state()
        self.restore_state(state)