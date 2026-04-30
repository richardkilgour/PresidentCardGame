"""
Lobby-state persistence across server restarts.

'Server restart' is simulated in-process: GamesKeeper is cleared and
load_all_games() reloads from games.json, exactly as the real startup
path does.  The live HTTP/socket server keeps running, so clients can
reconnect with the same session cookies and see no difference.

Test sequence (mirrors the scenario described in the task):
  1.  Owner creates game → restart → owner still at seat 0
  2.  Add Bot1 at seat 1   → restart → Bot1 still at seat 1
  3.  Add Bot2 at seat 2   → restart → Bots 1+2 still correct
  4.  Add Bot3 at seat 3   → restart → all three bots still correct
  5.  Start game → play some → restart mid-game → finish
"""
import json
import time

import pytest

from tests.dummy_client import DummyClient

BASE_URL = "http://localhost:5099"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_games_db():
    """
    Save and restore games.json around each test so stale games from
    previous runs (or prior tests in the same session) do not interfere.
    """
    from president.app.db import GAME_DB, load_data, save_data
    from president.app.game_keeper import GamesKeeper

    original = load_data(GAME_DB)
    save_data(GAME_DB, {})          # start each test with an empty DB
    GamesKeeper()._games.clear()    # and a clean in-memory state

    yield

    # After the test, restore the original DB and clear any test games
    save_data(GAME_DB, original)
    GamesKeeper()._games.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_restart():
    """Wipe in-memory game state and reload from disk."""
    from president.app.game_keeper import GamesKeeper
    from president.app.game_persistence import load_all_games
    GamesKeeper()._games.clear()
    load_all_games()


def _seat_names(game_id: str) -> list:
    """Return [name_at_seat0, …, name_at_seat3] for a given game."""
    from president.app.game_keeper import GamesKeeper
    return GamesKeeper().get_player_names(game_id)


def _game_exists(game_id: str) -> bool:
    from president.app.game_keeper import GamesKeeper
    return game_id in GamesKeeper().get_games()


def _disconnect_then_restart(client: DummyClient) -> None:
    """Disconnect the client, wait for the server to process it, then restart."""
    client.disconnect()
    time.sleep(0.15)      # give disconnect handler time to save
    _simulate_restart()


def _reconnect_and_settle(client: DummyClient) -> None:
    """Reconnect and wait for the rejoin handshake to complete."""
    client.reconnect()
    time.sleep(0.2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_owner_persists_across_restart():
    """
    Creating a game writes it to disk immediately.
    After a simulated restart the owner is still at seat 0 and is
    recognised as the game owner.
    """
    owner = DummyClient(BASE_URL)
    owner.setup()
    try:
        game_id = owner.create_game()
        assert game_id, "game was not created"

        # Verify initial in-memory state
        assert _seat_names(game_id)[0] == owner.username

        _disconnect_then_restart(owner)

        assert _game_exists(game_id),            "game lost after restart"
        names = _seat_names(game_id)
        assert names[0] == owner.username,       f"owner missing after restart: {names}"
        assert names[1] is None,                 f"seat 1 should be empty: {names}"
        assert names[2] is None,                 f"seat 2 should be empty: {names}"
        assert names[3] is None,                 f"seat 3 should be empty: {names}"

        _reconnect_and_settle(owner)
    finally:
        try:
            owner.quit_game()
        except Exception:
            pass
        owner.teardown()


def test_first_ai_player_persists_across_restart():
    """
    Owner adds an AI player at seat 1.
    After a restart seat 1 still holds that AI and no other seat has changed.
    """
    owner = DummyClient(BASE_URL)
    owner.setup()
    try:
        game_id = owner.create_game()
        assert game_id

        owner.client.add_ai_player(seat=1, name="Bot1", difficulty="Easy")
        time.sleep(0.1)

        _disconnect_then_restart(owner)

        assert _game_exists(game_id), "game lost after restart"
        names = _seat_names(game_id)
        assert names[0] == owner.username, f"owner missing: {names}"
        assert names[1] == "Bot1 (AI)",    f"Bot1 missing at seat 1: {names}"
        assert names[2] is None,           f"seat 2 should be empty: {names}"
        assert names[3] is None,           f"seat 3 should be empty: {names}"

        _reconnect_and_settle(owner)
    finally:
        try:
            owner.quit_game()
        except Exception:
            pass
        owner.teardown()


def test_second_ai_player_persists_across_restart():
    """
    Owner adds AI players at seats 1 and 2.
    After a restart between each addition, both seats are still correct.
    """
    owner = DummyClient(BASE_URL)
    owner.setup()
    try:
        game_id = owner.create_game()
        assert game_id

        # --- seat 1 ---
        owner.client.add_ai_player(seat=1, name="Bot1", difficulty="Easy")
        time.sleep(0.1)
        _disconnect_then_restart(owner)

        names = _seat_names(game_id)
        assert names[0] == owner.username, f"owner missing after 1st restart: {names}"
        assert names[1] == "Bot1 (AI)",    f"Bot1 missing after 1st restart: {names}"
        _reconnect_and_settle(owner)

        # --- seat 2 ---
        owner.client.add_ai_player(seat=2, name="Bot2", difficulty="Medium")
        time.sleep(0.1)
        _disconnect_then_restart(owner)

        names = _seat_names(game_id)
        assert _game_exists(game_id),      "game lost after 2nd restart"
        assert names[0] == owner.username, f"owner missing after 2nd restart: {names}"
        assert names[1] == "Bot1 (AI)",    f"Bot1 wrong after 2nd restart: {names}"
        assert names[2] == "Bot2 (AI)",    f"Bot2 missing after 2nd restart: {names}"
        assert names[3] is None,           f"seat 3 should be empty: {names}"

        _reconnect_and_settle(owner)
    finally:
        try:
            owner.quit_game()
        except Exception:
            pass
        owner.teardown()


def test_third_ai_player_persists_across_restart():
    """
    Owner fills seats 1, 2, and 3 with AI players, restarting after each.
    All three AI players are at the correct seats after every restart.
    """
    owner = DummyClient(BASE_URL)
    owner.setup()
    try:
        game_id = owner.create_game()
        assert game_id

        # --- seat 1 ---
        owner.client.add_ai_player(seat=1, name="Bot1", difficulty="Easy")
        time.sleep(0.1)
        _disconnect_then_restart(owner)

        names = _seat_names(game_id)
        assert names[0] == owner.username, f"owner missing after 1st restart: {names}"
        assert names[1] == "Bot1 (AI)",    f"Bot1 missing after 1st restart: {names}"
        _reconnect_and_settle(owner)

        # --- seat 2 ---
        owner.client.add_ai_player(seat=2, name="Bot2", difficulty="Medium")
        time.sleep(0.1)
        _disconnect_then_restart(owner)

        names = _seat_names(game_id)
        assert names[0] == owner.username, f"owner missing after 2nd restart: {names}"
        assert names[1] == "Bot1 (AI)",    f"Bot1 wrong after 2nd restart: {names}"
        assert names[2] == "Bot2 (AI)",    f"Bot2 missing after 2nd restart: {names}"
        _reconnect_and_settle(owner)

        # --- seat 3 ---
        owner.client.add_ai_player(seat=3, name="Bot3", difficulty="Hard")
        time.sleep(0.1)
        _disconnect_then_restart(owner)

        names = _seat_names(game_id)
        assert _game_exists(game_id),      "game lost after 3rd restart"
        assert names[0] == owner.username, f"owner missing after 3rd restart: {names}"
        assert names[1] == "Bot1 (AI)",    f"Bot1 wrong after 3rd restart: {names}"
        assert names[2] == "Bot2 (AI)",    f"Bot2 wrong after 3rd restart: {names}"
        assert names[3] == "Bot3 (AI)",    f"Bot3 missing after 3rd restart: {names}"

        _reconnect_and_settle(owner)
    finally:
        try:
            owner.quit_game()
        except Exception:
            pass
        owner.teardown()


def test_game_survives_mid_round_restart():
    """
    Full lobby setup (owner + 3 AI bots), game started, some cards played,
    server restarted mid-game, owner reconnects and the game plays to completion.

    The DummyClient accumulates notify_played_out events across reconnects, so
    wait_for_game_over() correctly detects completion regardless of when the
    restart occurred within the episode.
    """
    owner = DummyClient(BASE_URL)
    owner.setup()
    try:
        game_id = owner.create_game()
        assert game_id

        owner.client.add_ai_player(seat=1, name="Bot1", difficulty="Medium")
        owner.client.add_ai_player(seat=2, name="Bot2", difficulty="Medium")
        owner.client.add_ai_player(seat=3, name="Bot3", difficulty="Medium")
        time.sleep(0.1)

        started = owner.start_game()
        assert started, "game did not start"

        # Speed up AI so the game finishes well within the test timeout.
        # step_interval is not persisted to disk, so we must set it again
        # after each simulated restart.
        from president.app.game_keeper import GamesKeeper
        GamesKeeper().get_game(game_id).step_interval = 0.05

        # Let the game progress for a bit so the restart happens truly mid-game.
        time.sleep(0.5)

        _disconnect_then_restart(owner)

        assert _game_exists(game_id), "game lost after mid-game restart"

        # Restore fast speed on the newly-loaded wrapper (not persisted).
        GamesKeeper().get_game(game_id).step_interval = 0.05

        _reconnect_and_settle(owner)

        assert owner.wait_for_game_over(timeout=30), \
            "game did not finish after mid-game restart"
    finally:
        owner.teardown()
