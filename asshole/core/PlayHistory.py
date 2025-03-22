#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Keep enough information that the hand can be replayer later, or rewound
Stores cards played, players that have passed and players that have finished
Can be used to query the remaining cards that are somewhere, and the value of the highest unplayed card
"""
from asshole.core.Meld import Meld

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List

# Clear enum for event types
class EventType(Enum):
    MELD = auto()        # Player made a meld
    ROUND_WON = auto()   # Player won a round
    COMPLETE = auto()    # Player completed their turn
    WAITING = auto()     # Player is waiting for their turn

@dataclass
class GameEvent:
    player: Any          # Player reference
    event_type: EventType
    primary_data: Any    # Meld info, player index, or other primary data
    secondary_data: Any  # Remaining cards or other secondary data

class PlayHistory:
    def __init__(self):
        self._memory = []
        self._players = []
        # Scratch for running total of players
        self._passed_players = []
        self._finished_players = []

    def clear(self):
        self._memory = []
        self._players = []
        self._passed_players = []
        self._finished_players = []

    def add_play(self, player, meld: [Meld|int], remaining_cards):
        self._update_player_order(player)

        self._handle_skipped_players(player)

        if len(self._passed_players) == 3:
            # This is the new lead of a fresh hand.
            self._memory.append(GameEvent(
                player=player,
                event_type=EventType.ROUND_WON,
                primary_data=None,
                secondary_data=remaining_cards,
            ))
            # TODO: Other players go to considered 'waiting'
            self._passed_players = []

        # Remember the cards that were played
        self._memory.append(GameEvent(
            player=player,
            event_type=EventType.MELD,
            primary_data=meld,
            secondary_data=remaining_cards,
        ))

        if remaining_cards == 0:
            self._handle_player_finished(player)
            return
        if not meld.cards and player not in self._passed_players:
            self._passed_players.append(player)

    def _handle_player_finished(self, player):
        # player gets a position
        if player not in self._finished_players:
            self._finished_players.append(player)
        # If this is the Citizen, only the Asshole is left
        if len(self._finished_players) == 3:
            for p in self._players:
                if p not in self._finished_players:
                    self._finished_players.append(p)

    def _handle_skipped_players(self, player):
        # Has the expected player already passed or completed? We are not explicitly notified, so add that to the list
        expected_player = self._players[0]
        while expected_player != player:
            temp_remaining_cards = expected_player.report_remaining_cards()
            if expected_player in self._finished_players:
                # player has already finished
                self._memory.append(GameEvent(
                    player=expected_player,
                    event_type=EventType.COMPLETE,
                    primary_data=self._finished_players.index(expected_player),
                    secondary_data=0,
                ))
            elif expected_player in self._passed_players:
                # player has already passed
                self._memory.append(GameEvent(
                    player=expected_player,
                    event_type=EventType.MELD,
                    primary_data=Meld(),
                    secondary_data=temp_remaining_cards,
                ))
            else:
                # player is waiting
                self._memory.append(GameEvent(
                    player=expected_player,
                    event_type=EventType.WAITING,
                    primary_data=0,
                    secondary_data=temp_remaining_cards,
                ))
            self._update_player_order(expected_player)
            expected_player = self._players[0]

    def _update_player_order(self, player):
        # Move previous player to end of list
        if not self._players:
            self._players = [player]
        else:
            # Move previous player to end of list
            self._players.append(self._players.pop(0))
            if player not in self._players:
                # Add the new player to the start of player list
                self._players = [player] + self._players

    def get_highest_remaining(self):
        for x in range(13, 0, -1):
            if self.get_number_remaining(x):
                return x
        return -1

    def get_number_remaining(self, value):
        """Calculate the number of cards remaining for the given value"""
        # Start with total possible (2 jokers or 4 regular cards)
        starting_count = 2 if value == 13 else 4

        # Subtract cards that have been played
        played = sum(len(m[1]) for m in self._memory
                     if m[1] and m[1][0].get_value() == value)

        return starting_count - played

    def previous_plays_generator(self, start_players, ignore_latest=True):
        """
        Generate previous play indices in reverse chronological order.

        Args:
            start_players: List of players, with the previous player at index 0
            ignore_latest: Whether to ignore the most recent play in memory

        Yields:
            Integer indices representing previous plays (54 for pass, others for actual plays)
        """
        # Make a copy of players that we can modify
        current_players = start_players.copy()

        # Determine where to start in the memory
        memory = self._memory
        if ignore_latest:
            memory = memory[:-1]  # Skip the last entry

        # Iterate through memory in reverse order
        for move in reversed(memory):
            # Rotate players until we find the one who made this move
            while move.player != current_players[0]:
                # Not the expected player - Assume they passed
                yield Meld()  # Pass code
                # Bring the last player to the front
                current_players = [current_players[-1]] + current_players[:-1]

            # We found the player who made the move
            other_meld = move.primary_data
            yield other_meld
            # Bring the last player to the front
            current_players = [current_players[-1]] + current_players[:-1]


    def __str__(self):
        return ' '.join(c for m in self._memory if m[2] for c in m[2])

def main():
    # Run some tests
    from asshole.core.GameMaster import GameMaster
    from asshole.players.PlayerSimple import PlayerSimple
    from asshole.core.Episode import State
    from asshole.core.CardGameListener import CardGameListener

    gm = GameMaster()
    listener = CardGameListener()
    gm.add_listener(listener)
    gm.add_player(PlayerSimple(f'name_0'))
    gm.add_player(PlayerSimple(f'name_1'))
    gm.add_player(PlayerSimple(f'name_2'))
    gm.add_player(PlayerSimple(f'name_3'))
    gm.start(1)
    while not gm.step():
        if gm.episode.state == State.FINISHED:
            break
    print("=======  REPLAY  ==========")
    for event in listener.memory._memory:
        # Check based on the actual event type value
        if event.event_type.value in [EventType.MELD.value, EventType.WAITING.value]:
            print(f"{event.player} {event.primary_data} {event.secondary_data}")
        elif event.event_type.value == EventType.COMPLETE.value:
            print(f"{event.player} IS DONE RANK {event.primary_data}")
        elif event.event_type.value == EventType.ROUND_WON.value:
            print(f"{event.player} WON THE HAND {event.primary_data}")
    print("=======  BACKWARDS  ==========")
    for p in listener.memory.previous_plays_generator(listener.players, False):
        if p in [0, 1,2 ,3]:
            print(f"Out in position {p}")
        else:
            print(p)

if __name__ == "__main__":
    main()
