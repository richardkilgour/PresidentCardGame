#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GameCheckpoint saves and restores the full game state to/from a JSON file.

Designed for:
  - On-demand saves (testing, debugging, tournament replay)
  - Crash recovery
  - Reconnecting players who drop out

The checkpoint is fully self-contained JSON. A companion utility
(display_checkpoint) renders it in human-readable form without
needing to restore the full game.

What is saved:
  - Round number
  - Episode state machine state
  - Rankings so far (in order)
  - Each player: name, type, hand, seat position
  - Current melds (playfield)
  - Discard pile checksum (count only, for integrity verification)
"""
import json
import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from president.core.GameMaster import GameMaster


class GameCheckpoint:

    VERSION = 1  # Bump when format changes

    def __init__(self, game_master: "GameMaster") -> None:
        self.game_master = game_master

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save the current game state to a JSON file.

        Args:
            path: File path to write to. Will overwrite if exists.
        """
        path = Path(path)
        try:
            state = self._serialise()
            path.write_text(json.dumps(state, indent=2))
            logging.info(f"Checkpoint saved to {path}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint to {path}: {e}")
            raise

    def save_on_error(self, path: str | Path) -> None:
        """
        Save a checkpoint and include the current traceback.
        Call from an except block for crash recovery.

        Args:
            path: File path to write to.
        """
        path = Path(path)
        try:
            state = self._serialise()
            state["error"] = traceback.format_exc()
            path.write_text(json.dumps(state, indent=2))
            logging.error(f"Crash checkpoint saved to {path}")
        except Exception as e:
            logging.error(f"Failed to save crash checkpoint: {e}")

    def save_on_player_disconnect(self, player, path: str | Path) -> None:
        """
        Save a checkpoint when a player disconnects.

        Args:
            player: The player who disconnected.
            path: File path to write to.
        """
        path = Path(path)
        try:
            state = self._serialise()
            state["disconnected_player"] = player.name
            path.write_text(json.dumps(state, indent=2))
            logging.warning(
                f"Disconnect checkpoint saved for {player.name} to {path}"
            )
        except Exception as e:
            logging.error(f"Failed to save disconnect checkpoint: {e}")

    def _serialise(self) -> dict:
        """Build the full state dictionary."""
        gm = self.game_master
        episode = gm.episode

        # --- Players ---
        players = []
        for i, player in enumerate(gm.player_manager.players):
            if player is None:
                players.append(None)
                continue
            players.append({
                "seat": i,
                "name": player.name,
                "type": player.__class__.__name__,
                "hand": [c.get_index() for c in player._hand],
                "position_count": player.position_count,
            })

        # --- Episode state ---
        if episode is None:
            episode_data = None
        else:
            # Current melds — store card indices, None for unplayed ('␆' or empty)
            current_melds = []
            for m in episode.current_melds:
                if m and m != '␆' and m.cards:
                    current_melds.append([c.get_index() for c in m.cards])
                else:
                    current_melds.append(None)

            # Rankings so far
            ranks = [p.name for p in episode.ranks]

            # Active players
            active = [p.name for p in episode.active_players]

            # Discard checksum
            discard_count = len(gm.episode.card_handler.discard_pile)

            episode_data = {
                "state": episode.state.name,
                "ranks": ranks,
                "active_players": active,
                "current_melds": current_melds,
                "discard_count": discard_count,
            }

        return {
            "version": self.VERSION,
            "round_number": gm.round_number,
            "positions": [p.name for p in gm.positions] if gm.positions else [],
            "players": players,
            "episode": episode_data,
        }

    # -------------------------------------------------------------------------
    # Restore
    # -------------------------------------------------------------------------

    @staticmethod
    def restore(path: str | Path, game_master: "GameMaster",
                player_registry: dict[str, type]) -> None:
        """
        Restore game state from a checkpoint file.

        Args:
            path: Path to the checkpoint JSON file.
            game_master: A freshly initialised GameMaster to restore into.
            player_registry: Maps player type name to class, e.g.
                             {"PlayerSimple": PlayerSimple, "PlayerRL": PlayerRL}
                             Required to reconstruct player objects.
        """
        path = Path(path)
        state = json.loads(path.read_text())

        GameCheckpoint._check_version(state)

        gm = game_master

        # --- Restore players ---
        from president.core.PlayingCard import PlayingCard
        from president.core.Meld import Meld
        from president.core.Episode import Episode, State

        for player_data in state["players"]:
            if player_data is None:
                continue
            player_class = player_registry[player_data["type"]]
            player = player_class(player_data["name"])
            player.position_count = player_data["position_count"]
            for card_index in player_data["hand"]:
                player.card_to_hand(PlayingCard(card_index))
            gm.add_player(player, player_data["seat"])

        # --- Restore round number and positions ---
        gm.round_number = state["round_number"]
        player_by_name = {
            p.name: p for p in gm.player_manager.players if p
        }
        gm.positions = [
            player_by_name[name] for name in state["positions"]
        ]

        # --- Restore episode ---
        episode_data = state.get("episode")
        if episode_data:
            episode = Episode(
                gm.player_manager,
                [player_by_name[name] for name in episode_data["ranks"]],
                gm.deck,
                gm.listener_list,
            )
            episode.state = State[episode_data["state"]]
            episode.active_players = [
                player_by_name[name] for name in episode_data["active_players"]
            ]
            # Restore current melds
            for i, meld_data in enumerate(episode_data["current_melds"]):
                if meld_data is None:
                    episode.current_melds[i] = '␆'
                else:
                    meld = None
                    for card_index in meld_data:
                        meld = Meld(PlayingCard(card_index), meld)
                    episode.current_melds[i] = meld
            gm.episode = episode

            # Verify discard checksum
            actual = len(gm.player_manager.discarded_cards)
            expected = episode_data["discard_count"]
            if actual != expected:
                logging.warning(
                    f"Discard checksum mismatch: expected {expected}, got {actual}"
                )

        logging.info(f"Checkpoint restored from {path}")

    @staticmethod
    def _check_version(state: dict) -> None:
        v = state.get("version")
        if v != GameCheckpoint.VERSION:
            raise ValueError(
                f"Checkpoint version mismatch: file is v{v}, "
                f"code expects v{GameCheckpoint.VERSION}."
            )

    @staticmethod
    def stamped_path(stem: str) -> Path:
        """
        Generate a timestamped file path.
        e.g. stamped_path("crash") -> Path("crash_2026-03-18_14-23-05.json")
        """
        from datetime import datetime
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return Path(f"{stem}_{stamp}.json")

    # -------------------------------------------------------------------------
    # Human-readable display — no GameMaster needed
    # -------------------------------------------------------------------------

    @staticmethod
    def display(path: str | Path) -> str:
        """
        Render a checkpoint file as a human-readable string.
        Does not require a running game — reads the JSON directly.

        Args:
            path: Path to the checkpoint JSON file.

        Returns:
            Formatted string describing the saved state.
        """
        path = Path(path)
        state = json.loads(path.read_text())
        lines = []

        lines.append(f"=== Checkpoint v{state.get('version', '?')} ===")
        lines.append(f"Round:     {state['round_number']}")
        lines.append(f"Positions: {', '.join(state['positions']) or 'None (first episode)'}")

        lines.append("\n--- Players ---")
        for p in state["players"]:
            if p is None:
                lines.append("  [empty seat]")
                continue
            hand = _format_hand(p["hand"])
            score = (2 * p["position_count"][0] +
                     p["position_count"][1] -
                     p["position_count"][2] -
                     2 * p["position_count"][3])
            lines.append(
                f"  Seat {p['seat']}  {p['name']} ({p['type']})\n"
                f"          Hand: {hand}\n"
                f"          Positions: {p['position_count']}  Score: {score}"
            )

        ep = state.get("episode")
        if ep is None:
            lines.append("\n--- No active episode ---")
        else:
            lines.append(f"\n--- Episode ---")
            lines.append(f"  State:          {ep['state']}")
            lines.append(f"  Rankings so far: {', '.join(ep['ranks']) or 'None'}")
            lines.append(f"  Active players: {', '.join(ep['active_players']) or 'None'}")
            lines.append(f"  Discard count:  {ep['discard_count']}")
            lines.append("  Current melds:")
            for i, meld in enumerate(ep["current_melds"]):
                meld_str = _format_hand(meld) if meld else "—"
                lines.append(f"    Seat {i}: {meld_str}")

        if "disconnected_player" in state:
            lines.append(f"\n⚠  Saved due to disconnect: {state['disconnected_player']}")
        if "error" in state:
            lines.append(f"\n⚠  Saved due to crash:\n{state['error']}")

        return "\n".join(lines)


# -------------------------------------------------------------------------
# Module-level helpers
# -------------------------------------------------------------------------

def _format_hand(card_indices: list[int] | None) -> str:
    """Format a list of card indices as a readable string."""
    if not card_indices:
        return "(empty)"
    from president.core.PlayingCard import PlayingCard
    return "  ".join(str(PlayingCard(i)) for i in card_indices)
