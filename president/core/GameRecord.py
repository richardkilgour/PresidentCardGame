#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GameRecord stores the complete game trajectory instead of a single snapshot.

Every significant event (play, pass, hand start, episode end) is appended to
the trajectory.  This serves three purposes:

  - Restore: the last state-snapshot entry is used to resume the game,
    identical to the old GameCheckpoint behaviour.
  - ML training: each play/pass entry records the full hand at decision time
    plus the action taken; final positions can be back-filled at game end.
  - Integrity: the trajectory is a tamper-evident log; the current state must
    be consistent with its history.

Format version 2.  Version-1 files (GameCheckpoint) are read transparently.
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
    from president.core.PlayerRegistry import PlayerRegistry


class GameRecord(CardGameListener):

    VERSION = 2

    def __init__(self, game_master: "GameMaster",
                 game_id: str | None = None) -> None:
        super().__init__()
        self.game_master = game_master
        self.game_id = game_id or str(uuid.uuid4())
        self.started_at: str = _now()
        self.completed_at: str | None = None
        self.trajectory: list[dict] = []

    # -------------------------------------------------------------------------
    # CardGameListener hooks — lightweight entries for plays, full snapshots
    # for hand/episode boundaries (the restore points).
    # -------------------------------------------------------------------------

    def notify_play(self, player, meld):
        super().notify_play(player, meld)
        # Captured before cards are removed from the hand (see PlayHistory.add_play).
        self.trajectory.append({
            "type": "play",
            "player": player.name,
            "seat": self._seat_of(player),
            "hand": [c.get_index() for c in player._hand],
            "cards": [c.get_index() for c in meld.cards],
            "timestamp": _now(),
        })

    def notify_pass(self, player):
        super().notify_pass(player)
        self.trajectory.append({
            "type": "pass",
            "player": player.name,
            "seat": self._seat_of(player),
            "hand": [c.get_index() for c in player._hand],
            "timestamp": _now(),
        })

    def notify_played_out(self, player, rank):
        super().notify_played_out(player, rank)
        self.trajectory.append({
            "type": "played_out",
            "player": player.name,
            "seat": self._seat_of(player),
            "rank": rank,
            "timestamp": _now(),
        })

    def notify_hand_won(self, winner):
        super().notify_hand_won(winner)
        self.trajectory.append({
            "type": "hand_won",
            "player": winner.name,
            "timestamp": _now(),
        })

    def notify_hand_start(self):
        super().notify_hand_start()
        self._append_state("hand_start")

    def notify_cards_swapped(self, player_good, player_bad, num_cards,
                             cards_to_good=None, cards_to_bad=None):
        super().notify_cards_swapped(player_good, player_bad, num_cards,
                                     cards_to_good, cards_to_bad)
        self.trajectory.append({
            "type": "swap",
            "player_good": player_good.name,
            "player_bad": player_bad.name,
            "num_cards": num_cards,
            "timestamp": _now(),
        })

    def notify_episode_end(self, final_ranks, starting_ranks):
        names = [p.name for p in final_ranks]
        self._append_state("episode_end", meta={"ranks": names})

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _append_state(self, event: str, meta: dict | None = None) -> None:
        """Append a full state snapshot as a restore point."""
        from president.core.GameCheckpoint import GameCheckpoint
        snapshot = GameCheckpoint(self.game_master)._serialise()
        entry = {
            "type": "state",
            "event": event,
            "snapshot": snapshot,
            "timestamp": _now(),
        }
        if meta:
            entry.update(meta)
        self.trajectory.append(entry)

    def _seat_of(self, player) -> int:
        try:
            return self.game_master.player_manager.players.index(player)
        except ValueError:
            return -1

    def _last_snapshot(self) -> dict | None:
        for entry in reversed(self.trajectory):
            if entry.get("type") == "state":
                return entry["snapshot"]
        return None

    # -------------------------------------------------------------------------
    # Serialise / save
    # -------------------------------------------------------------------------

    def serialise(self) -> dict:
        return {
            "version": self.VERSION,
            "game_id": self.game_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "trajectory": self.trajectory,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        try:
            path.write_text(json.dumps(self.serialise(), indent=2))
            logging.info(f"GameRecord saved to {path}")
        except Exception as e:
            logging.error(f"Failed to save GameRecord to {path}: {e}")
            raise

    def save_on_error(self, path: str | Path) -> None:
        path = Path(path)
        try:
            data = self.serialise()
            data["error"] = traceback.format_exc()
            path.write_text(json.dumps(data, indent=2))
            logging.error(f"Crash record saved to {path}")
        except Exception as e:
            logging.error(f"Failed to save crash record: {e}")

    def mark_complete(self) -> None:
        self.completed_at = _now()

    def archive(self, archive_dir: str | Path) -> Path:
        archive_dir = Path(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        dest = archive_dir / f"{self.game_id}.json"
        dest.write_text(json.dumps(self.serialise(), indent=2))
        logging.info(f"GameRecord archived to {dest}")
        return dest

    # -------------------------------------------------------------------------
    # Restore
    # -------------------------------------------------------------------------

    @staticmethod
    def restore(path: str | Path, game_master: "GameMaster",
                player_registry: "PlayerRegistry") -> "GameRecord":
        path = Path(path)
        data = json.loads(path.read_text())
        record = GameRecord.restore_from_dict(data, game_master, player_registry)
        logging.info(f"GameRecord restored from {path}")
        return record

    @staticmethod
    def restore_from_dict(data: dict, game_master: "GameMaster",
                          player_registry: "PlayerRegistry") -> "GameRecord":
        from president.core.GameCheckpoint import GameCheckpoint
        version = data.get("version", 1)

        if version == 1:
            # Back-compat: wrap a v1 GameCheckpoint dict in a one-entry trajectory.
            GameCheckpoint.restore_from_dict(data, game_master, player_registry)
            record = GameRecord(game_master)
            record.trajectory = [
                {"type": "state", "event": "restored_v1",
                 "snapshot": data, "timestamp": _now()}
            ]
            return record

        if version != GameRecord.VERSION:
            raise ValueError(
                f"GameRecord version mismatch: file is v{version}, "
                f"code expects v{GameRecord.VERSION}."
            )

        # Find the most recent state snapshot to restore from.
        snapshot = None
        for entry in reversed(data["trajectory"]):
            if entry.get("type") == "state":
                snapshot = entry["snapshot"]
                break

        if snapshot is None:
            raise ValueError(
                "No restorable state snapshot found in GameRecord trajectory."
            )

        GameCheckpoint.restore_from_dict(snapshot, game_master, player_registry)

        record = GameRecord(game_master, game_id=data.get("game_id"))
        record.started_at = data.get("started_at", _now())
        record.completed_at = data.get("completed_at")
        record.trajectory = data["trajectory"]
        logging.info(
            f"GameRecord restored from dict "
            f"({len(record.trajectory)} trajectory entries)"
        )
        return record

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    @staticmethod
    def display(path: str | Path) -> str:
        from president.core.GameCheckpoint import GameCheckpoint, _format_snapshot
        path = Path(path)
        data = json.loads(path.read_text())
        version = data.get("version", 1)

        if version == 1:
            return GameCheckpoint.display(path)

        lines = [
            f"=== GameRecord v{version} ===",
            f"Game ID:   {data.get('game_id', '?')}",
            f"Started:   {data.get('started_at', '?')}",
            f"Completed: {data.get('completed_at') or '(in progress)'}",
            f"Entries:   {len(data['trajectory'])}",
        ]

        for entry in reversed(data["trajectory"]):
            if entry.get("type") == "state":
                lines.append("")
                lines.append(_format_snapshot(entry["snapshot"]))
                break

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def stamped_path(stem: str) -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return Path(f"{stem}_{stamp}.json")


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
