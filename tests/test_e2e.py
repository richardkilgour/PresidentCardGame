import time

from tests.dummy_client import DummyClient

BASE_URL = "http://localhost:5098"


def _start_game(n=4):
    """Create n DummyClients, log in, create/join a game, and start it."""
    clients = [DummyClient(BASE_URL) for _ in range(n)]
    for c in clients:
        c.setup()
    game_id = clients[0].create_game()
    for c in clients[1:]:
        c.join_game(game_id)
    time.sleep(0.2)
    clients[0].start_game()
    return clients, game_id


def _cleanup(clients):
    """Quit then disconnect every client so the server deletes the game."""
    for c in clients:
        try:
            c.quit_game()
        except Exception:
            pass
    for c in clients:
        try:
            c.teardown()
        except Exception:
            pass


def test_four_players_complete_a_game():
    clients, _ = _start_game()
    try:
        assert clients[0].wait_for_game_over(timeout=30), "Game did not complete in time"
    finally:
        _cleanup(clients)


def test_player_reconnects_after_disconnect():
    """A player who drops mid-game can reconnect before the AI-replacement timeout
    and resume playing without disrupting the game."""
    clients, _ = _start_game()
    try:
        time.sleep(0.5)  # let the game get going

        # Drop one player
        clients[1].disconnect()

        # Reconnect immediately with the same HTTP session (same username)
        time.sleep(0.1)
        clients[1].reconnect()

        assert clients[0].wait_for_game_over(timeout=30), "Game did not complete after reconnect"
    finally:
        _cleanup(clients)


def test_game_survives_server_restart():
    """Game state persisted to disk survives a server restart; all clients
    can rejoin and play to completion."""
    from president.app.game_keeper import GamesKeeper
    from president.app.game_persistence import load_all_games, save_game

    clients, game_id = _start_game()
    try:
        time.sleep(0.5)  # let the game get going

        # Persist and disconnect all players (disconnect triggers save_game too,
        # but we save explicitly first to capture a clean mid-game snapshot)
        save_game(game_id)
        for c in clients:
            c.disconnect()
        time.sleep(0.1)

        # Simulate server restart: wipe in-memory state, restore from disk
        GamesKeeper()._games.clear()
        load_all_games()

        # All clients reconnect — same HTTP sessions, server recognises each user,
        # re-adds them to the game room, and emits rejoin_game
        for c in clients:
            c.reconnect()
        time.sleep(0.2)

        assert clients[0].wait_for_game_over(timeout=30), "Game did not complete after restart"
    finally:
        _cleanup(clients)
