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
from typing import Any


# Clear enum for event types
class EventType(Enum):
    MELD = "meld"        # Player made a meld
    ROUND_WON = "round_won"   # Player won a round
    COMPLETE = "completed"    # Player completed their turn
    WAITING = "waiting"     # Player is waiting for their turn


@dataclass
class GameEvent:
    player: Any  # Player reference
    event_type: EventType
    meld: Any  # Meld info, player index, or other primary data
    remaining_cards: Any  # Remaining cards or other secondary data

    def __str__(self):
        return (f"{self.event_type}\t{self.player=}\t"
                f"{self.meld=}\t{self.remaining_cards=})")


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

    def add_play(self, player, meld: [Meld | int]):
        # Note: It has not been executed yet, so hand includes the meld
        remaining_cards = player.report_remaining_cards() - len(meld.cards)

        self._update_player_order(player)

        self._handle_skipped_players(player)

        if len(self._passed_players) == 3:
            self._handle_round_won(player, remaining_cards)

        # Remember the cards that were played
        self._memory.append(GameEvent(
            player=player,
            event_type=EventType.MELD,
            meld=meld,
            remaining_cards=remaining_cards,
        ))

        if remaining_cards == 0:
            self._handle_player_finished(player)
            return
        if not meld.cards and player not in self._passed_players:
            self._passed_players.append(player)

    def _handle_round_won(self, player, remaining_cards):
        # This is the new lead of a fresh hand.
        self._memory.append(GameEvent(
            player=player,
            event_type=EventType.ROUND_WON,
            meld=None,
            remaining_cards=remaining_cards,
        ))
        # Other players go to 'waiting'
        self._rotate_player_list()
        for _ in range(0, 3):
            self._memory.append(GameEvent(
                player=self._players[0],
                event_type=EventType.WAITING,
                meld=None,
                remaining_cards=self._players[0].report_remaining_cards()
            ))
            self._rotate_player_list()
        self._passed_players = []

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
                    meld=self._finished_players.index(expected_player),
                    remaining_cards=0,
                ))
            elif expected_player in self._passed_players:
                # player has already passed
                self._memory.append(GameEvent(
                    player=expected_player,
                    event_type=EventType.MELD,
                    meld=Meld(),
                    remaining_cards=temp_remaining_cards,
                ))
            else:
                # player is waiting
                self._memory.append(GameEvent(
                    player=expected_player,
                    event_type=EventType.WAITING,
                    meld=0,
                    remaining_cards=temp_remaining_cards,
                ))
            self._update_player_order(expected_player)
            expected_player = self._players[0]

    def _rotate_player_list(self):
        self._players.append(self._players.pop(0))

    def _update_player_order(self, player):
        # Move previous player to end of list
        if not self._players:
            self._players = [player]
        else:
            # Move previous player to end of list
            self._rotate_player_list()
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

    def previous_plays_generator(self, start_pos:int = 0):
        """
        Generate previous play indices in reverse chronological order.

        Args:
            start_pos: starting position in the memory, and work back from there Default: 0 - take the whole memory

        Yields:
            Tuple of:
               player name
               integer indices representing previous plays (54 for pass, others for actual plays)
               string representation of the event
        """
        # Determine where to start in the memory
        if start_pos <= 0:
            memory = self._memory
        else:
            memory = self._memory[:-start_pos] if start_pos < len(self._memory) else []  # Remove last n items or return empty list

        # Iterate through memory in reverse order
        for move in reversed(memory):
            # We have the move in question
            if move.event_type == EventType.MELD:
                yield move.player, move.meld, f"{move.player.name} PLAYED {move.meld}"
            if move.event_type == EventType.ROUND_WON:
                yield move.player, -1, f"{move.player.name} WON THE ROUND"
            if move.event_type == EventType.COMPLETE:
                yield move.player, move.meld, f"PLAYER IS FINISHED IN POS {move.meld}"
            if move.event_type == EventType.WAITING:
                yield move.player, None, f"{move.player.name} IS WAITING"
        # Three players are waiting at the start of history
        for i in range(0, 3):
            # Edge case: If this is the first payer, the other players may not have been added yet
            player = self._memory[3-i].player if len(self._memory) >= (4-i) else None
            yield player, None, f"{getattr(player, 'name', '<unknown>')} IS WAITING"

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
    print("=======  RAW MEMORY  ==========")
    for i, event in enumerate(listener.memory._memory):
        # Check based on the actual event type value
        if event.event_type.value in [EventType.MELD.value, EventType.WAITING.value]:
            print(f"{event.player} {event.meld} {event.remaining_cards}")
        elif event.event_type.value == EventType.COMPLETE.value:
            print(f"{event.player} IS DONE RANK {event.meld}")
        elif event.event_type.value == EventType.ROUND_WON.value:
            print(f"{event.player} WON THE HAND {event.meld}")
    print("=======  MEMORY WITH HISoTRY  ==========")
    for i in range(0, len(listener.memory._memory)):
        prev_plays = []
        for player, play, desc in listener.memory.previous_plays_generator(i):
            # Perpend the previous ones
            prev_plays.insert(0, desc + "\t")
            if len(prev_plays) >= 4:
                break
        print(f"{' '.join(prev_plays)}")

if __name__ == "__main__":
    main()
