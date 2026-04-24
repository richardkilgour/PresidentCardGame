import logging
from datetime import datetime, timezone

from president.app.db import GAME_DB, load_data, save_data
from president.app.game_keeper import GamesKeeper
from president.core.GameCheckpoint import GameCheckpoint
from president.core.PlayerRegistry import PlayerRegistry
from president.players.AsyncPlayer import AsyncPlayer
from president.players.PlayerHolder import PlayerHolder
from president.players.PlayerSimple import PlayerSimple
from president.players.PlayerSplitter import PlayerSplitter


def _make_restore_registry() -> PlayerRegistry:
    registry = PlayerRegistry()
    registry.register(AsyncPlayer, name="AsyncPlayer")
    registry.register(PlayerSimple, name="PlayerSimple")
    registry.register(PlayerHolder, name="PlayerHolder")
    registry.register(PlayerSplitter, name="PlayerSplitter")
    return registry


def save_game(game_id: str) -> None:
    """Serialise the current game state to games.json."""
    game = GamesKeeper().get_game(game_id)
    checkpoint = GameCheckpoint(game)
    state = checkpoint._serialise()
    games_data = load_data(GAME_DB)
    games_data[game_id] = {
        "checkpoint": state,
        "reserved_slots": {str(k): v for k, v in game.reserved_slots.items()},
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    save_data(GAME_DB, games_data)


def delete_game(game_id: str) -> None:
    """Remove a game from games.json (called when a game ends or is archived)."""
    games_data = load_data(GAME_DB)
    if game_id in games_data:
        del games_data[game_id]
        save_data(GAME_DB, games_data)


def load_all_games() -> None:
    """
    Restore all persisted active games into GamesKeeper on server startup.
    Corrupted or unrestorable games are removed from disk.
    """
    from president.app.game_event_handler import GameEventHandler
    from president.app.extensions import socketio
    from president.app.game_wrapper import GameWrapper

    games_data = load_data(GAME_DB)
    if not games_data:
        return

    registry = _make_restore_registry()
    failed = []

    for game_id, data in games_data.items():
        try:
            wrapper = GameWrapper(game_id, GameEventHandler(socketio, game_id))
            GameCheckpoint.restore_from_dict(data["checkpoint"], wrapper, registry)
            wrapper.reserved_slots = {
                int(k): v for k, v in data.get("reserved_slots", {}).items()
            }
            GamesKeeper().add_game(game_id, wrapper)
            logging.info(f"Restored game {game_id}")
        except Exception as e:
            logging.error(f"Failed to restore game {game_id}: {e}")
            failed.append(game_id)

    if failed:
        games_data = load_data(GAME_DB)
        for game_id in failed:
            games_data.pop(game_id, None)
        save_data(GAME_DB, games_data)
