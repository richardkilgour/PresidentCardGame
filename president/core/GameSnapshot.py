#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GameSnapshot provides a read-only view of the current game state for rendering.

Assembled on demand by calling GameSnapshot.from_game(). Does not hold
any mutable references — it is a pure data object representing a moment in time.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from president.core.Meld import Meld


class PlayerStatus(Enum):
    ABSENT  = "Absent"   # Not in the game
    WAITING = "Waiting"  # In the hand, hasn't played yet
    PLAYED  = "Played"   # Has played a meld this round
    PASSED  = "Passed"   # Has passed this round
    FINISHED = "Finished" # Played out, has a rank


@dataclass(frozen=True)
class PlayerSnapshot:
    name: str
    status: PlayerStatus
    hand_size: int
    last_meld: Meld | None      # None if waiting, passed, or finished
    rank: int | None            # 0=President .. 3=Scumbag, None if still playing
    hand: list | None           # Only populated for the client's own player


@dataclass(frozen=True)
class GameSnapshot:
    players: list[PlayerSnapshot]
    episode_state: Any          # Episode State enum value
    round_number: int

    @staticmethod
    def from_game(game_master, client_player=None) -> "GameSnapshot":
        """
        Assemble a snapshot from the current game state.

        Args:
            game_master: The GameMaster instance to snapshot.
            client_player: The player whose hand should be included in full.
                           All other players' hands are hidden (hand=None).
        """
        episode = game_master.episode
        player_snapshots = []

        for i, player in enumerate(game_master.player_manager.players):
            if player is None:
                player_snapshots.append(PlayerSnapshot(
                    name="",
                    status=PlayerStatus.ABSENT,
                    hand_size=0,
                    last_meld=None,
                    rank=None,
                    hand=None,
                ))
                continue

            # Determine status
            if episode is None:
                status = PlayerStatus.WAITING
                last_meld = None
                rank = None
            elif player in episode.ranks:
                status = PlayerStatus.FINISHED
                last_meld = None
                rank = episode.ranks.index(player)
            elif player not in episode.active_players:
                status = PlayerStatus.PASSED
                last_meld = None
                rank = None
            else:
                meld = episode.current_melds[i]
                if meld and meld != '␆':
                    status = PlayerStatus.PLAYED
                    last_meld = meld
                else:
                    status = PlayerStatus.WAITING
                    last_meld = None
                rank = None

            player_snapshots.append(PlayerSnapshot(
                name=player.name,
                status=status,
                hand_size=player.report_remaining_cards(),
                last_meld=last_meld,
                rank=rank,
                hand=list(player._hand) if player is client_player else None,
            ))

        return GameSnapshot(
            players=player_snapshots,
            episode_state=episode.state if episode else None,
            round_number=game_master.round_number,
        )