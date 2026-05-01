#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GameSave persists game-level state between episodes.

Written once per completed episode. Contains player identities, seating,
cumulative scores, and game parameters. Holds no within-episode state —
that is EpisodeSave's responsibility.

Restoring from GameSave puts the game at "ready to start the next episode".
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from president.core.GameMaster import GameMaster
    from president.core.PlayerRegistry import PlayerRegistry

logger = logging.getLogger(__name__)


class GameSave:

    VERSION = 1

    def __init__(self, game_master: "GameMaster") -> None:
        self.game_master = game_master

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self._serialise(), indent=2))
        logger.info(f"GameSave written to {path}")

    def _serialise(self) -> dict:
        gm = self.game_master
        players = []
        for i, player in enumerate(gm.player_manager.players):
            if player is None:
                players.append(None)
                continue
            players.append({
                "seat": i,
                "name": player.name,
                "type": player.__class__.__name__,
                "position_count": player.position_count,
            })
        return {
            "version": self.VERSION,
            "round_number": gm.round_number,
            "total_rounds": gm.number_of_rounds,
            "positions": [p.name for p in gm.positions] if gm.positions else [],
            "players": players,
            "parameters": {
                "policy": gm.policy.name,
                "fallback_player_name": gm.fallback_player_name,
            },
        }

    # -------------------------------------------------------------------------
    # Restore
    # -------------------------------------------------------------------------

    @staticmethod
    def restore(path: str | Path, game_master: "GameMaster",
                player_registry: "PlayerRegistry") -> None:
        data = json.loads(Path(path).read_text())
        GameSave.restore_from_dict(data, game_master, player_registry)
        logger.info(f"GameSave restored from {path}")

    @staticmethod
    def restore_from_dict(data: dict, game_master: "GameMaster",
                          player_registry: "PlayerRegistry") -> None:
        if data.get("version") != GameSave.VERSION:
            raise ValueError(
                f"GameSave version mismatch: file is v{data.get('version')}, "
                f"code expects v{GameSave.VERSION}."
            )
        gm = game_master
        for player_data in data["players"]:
            if player_data is None:
                continue
            player = player_registry.create(player_data["type"], player_data["name"])
            player.position_count = player_data["position_count"]
            gm.add_player(player, player_data["seat"])

        gm.round_number = data["round_number"]
        gm.number_of_rounds = data["total_rounds"]
        player_by_name = {p.name: p for p in gm.player_manager.players if p}
        gm.positions = [player_by_name[name] for name in data["positions"]]
        logger.info("GameSave restored from dict")

    # -------------------------------------------------------------------------
    # Combined save/restore (single file for offline use)
    # -------------------------------------------------------------------------

    def save_combined(self, episode_save: "EpisodeSave",
                      path: str | Path) -> None:
        """Write game state and episode trajectory to a single file."""
        from president.core.EpisodeSave import EpisodeSave
        data = {
            "game_save": self._serialise(),
            "episode_save": episode_save.serialise(),
        }
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info(f"Combined save written to {path}")

    @staticmethod
    def restore_combined(path: str | Path, game_master: "GameMaster",
                         player_registry: "PlayerRegistry") -> "EpisodeSave":
        """Restore game and episode state from a combined save file."""
        from president.core.EpisodeSave import EpisodeSave
        data = json.loads(Path(path).read_text())
        GameSave.restore_from_dict(data["game_save"], game_master, player_registry)
        record = EpisodeSave.restore_from_dict(data["episode_save"], game_master)
        logger.info(f"Combined save restored from {path}")
        return record

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    @staticmethod
    def display(path: str | Path) -> str:
        """Display a GameSave or combined save file in human-readable form."""
        from president.core.EpisodeSave import _format_episode_save
        data = json.loads(Path(path).read_text())
        if "game_save" in data:
            parts = [_format_game_save(data["game_save"])]
            if data.get("episode_save"):
                parts.append(_format_episode_save(data["episode_save"]))
            return "\n\n".join(parts)
        return _format_game_save(data)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def stamped_path(stem: str, directory: str = "../saves/") -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return Path(f"{directory}{stem}_{stamp}.json")


def _format_game_save(data: dict) -> str:
    lines = [
        f"=== GameSave v{data.get('version', '?')} ===",
        f"Round:     {data.get('round_number', '?')} / "
        f"{data.get('total_rounds') or '∞'}",
        f"Positions: {', '.join(data.get('positions', [])) or 'None (first episode)'}",
        "",
        "--- Players ---",
    ]
    for p in data.get("players", []):
        if p is None:
            lines.append("  [empty seat]")
            continue
        pc = p["position_count"]
        score = 2 * pc[0] + pc[1] - pc[2] - 2 * pc[3]
        lines.append(
            f"  Seat {p['seat']}  {p['name']} ({p['type']})"
            f"  positions={pc}  score={score}"
        )
    return "\n".join(lines)
