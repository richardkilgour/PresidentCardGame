#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Keep enough information that the hand can be replayed later, or rewound.
Stores cards played, players that have passed and players that have finished.
Can be used to query the remaining cards that are somewhere, and the value of the highest unplayed card.
"""
from __future__ import annotations

from president.core.Meld import Meld

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EventType(Enum):
    MELD = "meld"           # Player made a meld (includes passes)
    ROUND_WON = "round_won" # Player won a round
    COMPLETE = "completed"  # Player completed their turn (played out)
    WAITING = "waiting"     # Player is waiting (sitting out a new hand)


@dataclass
class GameEvent:
    player: Any
    event_type: EventType
    meld: Any               # Meld for MELD events; rank index for COMPLETE; None otherwise
    remaining_cards: int    # Hand size after the action
    hand: list | None = None  # Full hand contents at decision time, acting player only

    def __str__(self):
        return (f"{self.event_type}\t{self.player=}\t"
                f"{self.meld=}\t{self.remaining_cards=}")


class PlayHistory:
    def __init__(self):
        self._memory = []
        self._finished_players = []
        self._players = []             # Seated order for clockwise validation
        self._last_play_player = None  # Last player to act (meld or pass)
        self._starting_positions = {}  # player → rank index at episode start (empty = neutral)
        self._final_positions = {}     # player → rank index at episode end

    def clear(self):
        self._memory = []
        self._finished_players = []
        self._last_play_player = None
        # _players is preserved — seating doesn't change between hands

    def set_players(self, players):
        """Record the seated player order (excluding empty seats)."""
        self._players = [p for p in players if p is not None]

    def _get_expected_next_player(self):
        """Return the one valid next player: first clockwise from last who hasn't passed or finished."""
        if not self._players or self._last_play_player is None:
            return None
        if self._last_play_player not in self._players:
            return None
        # Collect players who passed in the current round (since last ROUND_WON)
        passed_this_round = set()
        for event in reversed(self._memory):
            if event.event_type == EventType.ROUND_WON:
                break
            if event.event_type == EventType.MELD and event.meld is not None and not event.meld.cards:
                passed_this_round.add(event.player)
        finished = set(self._finished_players)
        n = len(self._players)
        last_idx = self._players.index(self._last_play_player)
        for i in range(1, n):
            candidate = self._players[(last_idx + i) % n]
            if candidate not in passed_this_round and candidate not in finished:
                return candidate
        return None

    def add_play(self, player, meld: Meld):
        """Record a meld or pass. Validates clockwise order. Called before cards are removed."""
        expected = self._get_expected_next_player()
        if expected is not None and player is not expected:
            raise ValueError(
                f"Clockwise order violation: expected "
                f"{getattr(expected, 'name', expected)} but got "
                f"{getattr(player, 'name', player)}"
            )
        self._last_play_player = player

        remaining_cards = player.report_remaining_cards() - len(meld)
        self._memory.append(GameEvent(
            player=player,
            event_type=EventType.MELD,
            meld=meld,
            remaining_cards=remaining_cards,
            hand=list(player._hand),  # captured before cards are removed
        ))
        if remaining_cards == 0:
            self._handle_player_finished(player)

    def add_round_won(self, player):
        """Record that a player has won the round and becomes the new lead."""
        self._last_play_player = None  # Reset: winner may start the next round
        self._memory.append(GameEvent(
            player=player,
            event_type=EventType.ROUND_WON,
            meld=None,
            remaining_cards=player.report_remaining_cards()
        ))
        self._passed_players = []

    def add_waiting(self, player):
        """Record that a player is waiting (sitting out a new hand)."""
        self._memory.append(GameEvent(
            player=player,
            event_type=EventType.WAITING,
            meld=None,
            remaining_cards=player.report_remaining_cards()
        ))

    def record_swap(self, player_good, player_bad, num_cards):
        """
        Infer and record starting positions from a card-swap notification.

        The swap with the most cards (num_cards == n // 2) is always the first
        notification of a new episode, so it resets both position dicts.

        Rank inference (mirrors Episode.swap_cards):
            rank_good = n // 2 - num_cards   (0 = President, 1 = VP, …)
            rank_bad  = n - 1 - n // 2 + num_cards   (n-1 = Scumbag, …)
        """
        n = len(self._players)
        if num_cards == n // 2:          # First swap of a new episode
            self._starting_positions = {}
            self._final_positions = {}
        rank_good = n // 2 - num_cards
        rank_bad  = n - 1 - n // 2 + num_cards
        self._starting_positions[player_good] = rank_good
        self._starting_positions[player_bad]  = rank_bad

    def starting_position(self, player) -> int | None:
        """Return the player's rank index at the start of this episode, or None if neutral."""
        return self._starting_positions.get(player)

    def final_position(self, player) -> int | None:
        """Return the player's rank index at the end of this episode, or None if not yet finished."""
        return self._final_positions.get(player)

    def add_player_finished(self, player, rank):
        """Record that a player has played out and assign their rank."""
        self._memory.append(GameEvent(
            player=player,
            event_type=EventType.COMPLETE,
            meld=rank,
            remaining_cards=0,
        ))
        self._final_positions[player] = rank
        self._handle_player_finished(player)

    def _handle_player_finished(self, player):
        if player not in self._finished_players:
            self._finished_players.append(player)

    def reconstruct_hand(self, player, existing_cards=None):
        """
        Reconstruct a player's starting hand from the play history.

        Add back every meld the player played to existing_cards (which
        represents cards they still hold — [] for everyone except the
        scumbag, whose leftover cards should be passed in before they
        are discarded).

        Returns:
            Sorted list of PlayingCard matching the player's original hand.
        """
        cards = list(existing_cards) if existing_cards else []
        for event in self._memory:
            if (event.player is player
                    and event.event_type == EventType.MELD
                    and event.meld is not None
                    and event.meld.cards):
                cards.extend(event.meld.cards)
        cards.sort(key=lambda c: c.get_index())
        return cards

    def get_highest_remaining(self):
        """Return the value of the highest card not yet played, or -1 if none."""
        for x in range(13, 0, -1):
            if self.get_number_remaining(x):
                return x
        return -1

    def current_target(self) -> Meld | None:
        """
        Return the meld that must currently be beaten, or None for an opening lead.

        Scans backwards through memory and returns the most recent non-pass meld,
        stopping at the last ROUND_WON boundary.
        """
        for event in reversed(self._memory):
            if event.event_type == EventType.ROUND_WON:
                return None
            if event.event_type == EventType.MELD and event.meld and event.meld.cards:
                return event.meld
        return None

    def last_event_for(self, player) -> GameEvent | None:
        """
        Return the most recent non-waiting event for a player.
        Returns None if the player has not yet acted this hand.
        """
        for event in reversed(self._memory):
            if event.player is player and event.event_type != EventType.WAITING:
                return event
        return None

    def get_number_remaining(self, value):
        """Calculate the number of unplayed cards of the given value."""
        starting_count = 2 if value == 13 else 4
        played = sum(
            len(m.meld)
                for m in self._memory
                    if m.event_type == EventType.MELD
                    and m.meld
                    and m.meld.cards
                    and m.meld.cards[0].get_value() == value
        )
        return starting_count - played

    def previous_plays_generator(self, plays_to_skip: int = 0):
        """
        Generate events in reverse chronological order.

        Args:
            plays_to_skip: Number of most recent events to skip. Default 0.

        Yields:
            Tuple of (player, meld_or_code, description_string)
            meld_or_code is:
                Meld   for MELD events
                -1     for ROUND_WON
                int    rank for COMPLETE
                None   for WAITING
        """
        if plays_to_skip <= 0:
            memory = self._memory
        else:
            memory = self._memory[:-plays_to_skip] if plays_to_skip < len(self._memory) else []

        for move in reversed(memory):
            name = getattr(move.player, 'name', '<unknown>')
            if move.event_type == EventType.MELD:
                yield move.player, move.meld, f"{name} PLAYED {move.meld}"
            elif move.event_type == EventType.ROUND_WON:
                yield move.player, -1, f"{name} WON THE ROUND"
            elif move.event_type == EventType.COMPLETE:
                yield move.player, move.meld, f"PLAYER IS FINISHED IN POS {move.meld}"
            elif move.event_type == EventType.WAITING:
                yield move.player, None, f"{name} IS WAITING"

        # Three players are waiting at the start of history
        for i in range(0, 3):
            player = self._memory[3 - i].player if len(self._memory) >= (4 - i) else None
            yield player, None, f"{getattr(player, 'name', '<unknown>')} IS WAITING"
