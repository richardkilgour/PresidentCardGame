#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EpisodeSave records the full event trajectory of a single episode.

Written incrementally after each event, so a crash at any point leaves a
valid partial trajectory. On restore, replaying the trajectory reconstructs
both player hands and PlayHistory memory exactly — no snapshots needed.

Serves: crash recovery, game replay, reinforcement learning training.

Format v1. Links to a GameSave via game_id.

Event sequence within an episode:
  swap*       — card exchanges before the first round (zero or more)
  deal        — each player's starting hand (post-swap), one per player
  play|pass   — per turn
  round_won   — end of each round
  complete    — each player as they finish (rank assigned)
  episode_end — final rankings
"""
from __future__ import annotations

import json
import logging
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from president.core.CardGameListener import CardGameListener

if TYPE_CHECKING:
    from president.core.GameMaster import GameMaster

logger = logging.getLogger(__name__)


class EpisodeSave(CardGameListener):

    VERSION = 1

    def __init__(self, game_master: "GameMaster",
                 game_id: str | None = None) -> None:
        super().__init__()
        self.game_master = game_master
        self.game_id = game_id or str(uuid.uuid4())
        self._path: Path | None = None
        self._is_first_hand: bool = True

        # Set at episode start
        self.round_number: int | None = None
        self.starting_ranks: list[str] | None = None
        self.events: list[dict] = []

    def set_path(self, path: str | Path) -> None:
        """Set save path for incremental flushing. Call before the episode starts."""
        self._path = Path(path)

    def reset_for_episode(self) -> None:
        """Clear event log for a new episode. Call before each episode begins."""
        self.events = []
        self._is_first_hand = True
        self.round_number = None
        self.starting_ranks = None

    # -------------------------------------------------------------------------
    # CardGameListener hooks
    # -------------------------------------------------------------------------

    def notify_cards_swapped(self, player_good, player_bad, num_cards,
                              cards_to_good=None, cards_to_bad=None) -> None:
        super().notify_cards_swapped(player_good, player_bad, num_cards,
                                     cards_to_good, cards_to_bad)
        self._append({
            "type": "swap",
            "player_good": player_good.name,
            "player_bad": player_bad.name,
            "cards_to_good": [c.get_index() for c in (cards_to_good or [])],
            "cards_to_bad":  [c.get_index() for c in (cards_to_bad  or [])],
        })

    def notify_hand_start(self) -> None:
        super().notify_hand_start()
        if not self._is_first_hand:
            return
        self._is_first_hand = False
        self.round_number = self.game_master.round_number
        # Capture each player's hand after dealing and swapping.
        for i, player in enumerate(self.game_master.player_manager.players):
            if player is not None:
                self._append({
                    "type": "deal",
                    "seat": i,
                    "player": player.name,
                    "cards": [c.get_index() for c in player._hand],
                })

    def notify_play(self, player, meld) -> None:
        super().notify_play(player, meld)
        self._append({
            "type": "play",
            "seat": self._seat(player),
            "player": player.name,
            "hand": [c.get_index() for c in player._hand],
            "cards": [c.get_index() for c in meld.cards],
        })

    def notify_pass(self, player) -> None:
        super().notify_pass(player)
        self._append({
            "type": "pass",
            "seat": self._seat(player),
            "player": player.name,
            "hand": [c.get_index() for c in player._hand],
        })

    def notify_hand_won(self, winner) -> None:
        super().notify_hand_won(winner)
        self._append({
            "type": "round_won",
            "player": winner.name,
        })

    def notify_played_out(self, player, rank) -> None:
        super().notify_played_out(player, rank)
        self._append({
            "type": "complete",
            "player": player.name,
            "rank": rank,
        })

    def notify_episode_end(self, final_ranks, starting_ranks) -> None:
        self.starting_ranks = [p.name for p in starting_ranks]
        self._append({
            "type": "episode_end",
            "final_ranks":   [p.name for p in final_ranks],
            "starting_ranks": self.starting_ranks,
        })

    # -------------------------------------------------------------------------
    # Serialise / save
    # -------------------------------------------------------------------------

    def serialise(self) -> dict:
        return {
            "version": self.VERSION,
            "game_id": self.game_id,
            "round_number": self.round_number,
            "starting_ranks": self.starting_ranks,
            "events": self.events,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.serialise(), indent=2))
        logger.info(f"EpisodeSave written to {path}")

    def save_on_error(self, path: str | Path) -> None:
        data = self.serialise()
        data["error"] = traceback.format_exc()
        Path(path).write_text(json.dumps(data, indent=2))
        logger.error(f"Crash record saved to {path}")

    def mark_complete(self) -> None:
        """No-op — episode is complete when it has an episode_end event."""
        pass

    def archive(self, archive_dir: str | Path) -> Path:
        archive_dir = Path(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        dest = archive_dir / f"{self.game_id}.json"
        dest.write_text(json.dumps(self.serialise(), indent=2))
        logger.info(f"EpisodeSave archived to {dest}")
        return dest

    # -------------------------------------------------------------------------
    # Restore
    # -------------------------------------------------------------------------

    @staticmethod
    def restore(path: str | Path,
                game_master: "GameMaster") -> "EpisodeSave":
        """
        Restore an in-progress episode from a trajectory file.
        GameSave.restore() must be called first to seat the players.
        """
        data = json.loads(Path(path).read_text())
        record = EpisodeSave.restore_from_dict(data, game_master)
        logger.info(f"EpisodeSave restored from {path}")
        return record

    @staticmethod
    def restore_from_dict(data: dict,
                          game_master: "GameMaster") -> "EpisodeSave":
        """
        Replay the event trajectory to reconstruct player hands, PlayHistory
        memory, and Episode state. GameSave must have been restored first.
        """
        if data.get("version") != EpisodeSave.VERSION:
            raise ValueError(
                f"EpisodeSave version mismatch: file is v{data.get('version')}, "
                f"code expects v{EpisodeSave.VERSION}."
            )

        from president.core.CardHandler import CardHandler
        from president.core.Episode import Episode, State
        from president.core.Meld import Meld
        from president.core.PlayingCard import PlayingCard
        from president.core.PlayHistory import EventType, GameEvent

        gm = game_master
        events = data.get("events", [])
        players = [p for p in gm.player_manager.players if p is not None]
        player_by_name = {p.name: p for p in players}
        seats = gm.player_manager.players      # list of 4, may contain None

        finished: list[tuple] = []             # (player, rank) in order

        for event in events:
            etype = event["type"]

            if etype == "deal":
                player = player_by_name.get(event["player"])
                if player is None:
                    continue
                player._hand.clear()
                for idx in event["cards"]:
                    player.card_to_hand(PlayingCard(idx))

            elif etype in ("play", "pass"):
                player = player_by_name.get(event["player"])
                if player is None:
                    continue

                hand_snapshot = [PlayingCard(idx) for idx in event["hand"]]

                if etype == "play":
                    meld = Meld()
                    meld.cards = [PlayingCard(idx) for idx in event["cards"]]
                    remaining = len(event["hand"]) - len(event["cards"])
                    played = set(event["cards"])
                    player._hand = [c for c in player._hand
                                    if c.get_index() not in played]
                else:
                    meld = Meld()          # empty Meld = pass
                    remaining = len(event["hand"])

                ge = GameEvent(
                    player=player,
                    event_type=EventType.MELD,
                    meld=meld,
                    remaining_cards=remaining,
                    hand=hand_snapshot,
                )
                for p in players:
                    p.memory._memory.append(ge)
                    p.memory._last_play_player = player

            elif etype == "round_won":
                player = player_by_name.get(event["player"])
                if player is None:
                    continue
                ge = GameEvent(
                    player=player,
                    event_type=EventType.ROUND_WON,
                    meld=None,
                    remaining_cards=len(player._hand),
                )
                for p in players:
                    p.memory._memory.append(ge)
                    p.memory._last_play_player = None

            elif etype == "complete":
                player = player_by_name.get(event["player"])
                rank = event["rank"]
                if player is None:
                    continue
                ge = GameEvent(
                    player=player,
                    event_type=EventType.COMPLETE,
                    meld=rank,
                    remaining_cards=0,
                )
                for p in players:
                    p.memory._memory.append(ge)
                    p.memory._final_positions[player] = rank
                    if player not in p.memory._finished_players:
                        p.memory._finished_players.append(player)
                finished.append((player, rank))

        # --- Reconstruct Episode ---
        starting_ranks_names = data.get("starting_ranks") or []
        starting_ranks = [player_by_name[n] for n in starting_ranks_names
                          if n in player_by_name]

        episode = Episode(
            gm.player_manager,
            starting_ranks,
            gm.deck,
            gm.listener_list,
            CardHandler(gm.deck),
        )
        episode.state = State.PLAYING
        episode.ranks = [p for p, _ in sorted(finished, key=lambda x: x[1])]
        episode.active_players = [p for p in players if p not in episode.ranks]
        episode.current_melds = _restore_current_melds(events, seats)
        episode.card_handler.discard_pile = _restore_discard(events)
        gm.episode = episode

        # Return a new EpisodeSave ready to continue appending
        record = EpisodeSave(gm, game_id=data.get("game_id"))
        record.round_number = data.get("round_number")
        record.starting_ranks = data.get("starting_ranks")
        record.events = list(events)
        record._is_first_hand = False
        logger.info(
            f"EpisodeSave restored from dict ({len(events)} events)"
        )
        return record

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    @staticmethod
    def display(path: str | Path) -> str:
        data = json.loads(Path(path).read_text())
        return _format_episode_save(data)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def stamped_path(stem: str, directory: str = "../saves/") -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return Path(f"{directory}{stem}_{stamp}.json")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _append(self, event: dict) -> None:
        event["ts"] = _now()
        self.events.append(event)
        if self._path:
            self._path.write_text(json.dumps(self.serialise(), indent=2))

    def _seat(self, player) -> int:
        try:
            return self.game_master.player_manager.players.index(player)
        except ValueError:
            return -1


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _restore_current_melds(events: list[dict], seats: list) -> list:
    """Reconstruct current_melds (per-seat last play in the current round)."""
    from president.core.Meld import Meld
    from president.core.PlayingCard import PlayingCard

    n = len(seats)
    current_melds: list = ['␆'] * n

    for event in events:
        if event["type"] == "round_won":
            current_melds = ['␆'] * n          # reset at each new round
        elif event["type"] == "play":
            seat = event.get("seat", -1)
            if 0 <= seat < n:
                meld = Meld()
                meld.cards = [PlayingCard(idx) for idx in event["cards"]]
                current_melds[seat] = meld
        elif event["type"] == "pass":
            seat = event.get("seat", -1)
            if 0 <= seat < n:
                current_melds[seat] = Meld()    # empty Meld = pass

    return current_melds


def _restore_discard(events: list[dict]) -> list:
    """Reconstruct the discard pile from all played cards."""
    from president.core.PlayingCard import PlayingCard
    discard = []
    for event in events:
        if event["type"] == "play":
            discard.extend(PlayingCard(idx) for idx in event["cards"])
    return discard


def _format_episode_save(data: dict) -> str:
    events = data.get("events", [])
    play_count = sum(1 for e in events if e["type"] == "play")
    pass_count = sum(1 for e in events if e["type"] == "pass")
    lines = [
        f"=== EpisodeSave v{data.get('version', '?')} ===",
        f"Game ID:   {data.get('game_id', '?')}",
        f"Round:     {data.get('round_number', '?')}",
        f"Starting:  {', '.join(data.get('starting_ranks') or []) or 'unknown'}",
        f"Events:    {len(events)}  ({play_count} plays, {pass_count} passes)",
    ]
    ep_end = next((e for e in reversed(events) if e["type"] == "episode_end"), None)
    if ep_end:
        lines.append(f"Final:     {', '.join(ep_end.get('final_ranks', []))}")
    else:
        lines.append("(episode in progress)")
    if data.get("error"):
        lines.append(f"\n⚠ Saved due to crash:\n{data['error']}")
    return "\n".join(lines)
