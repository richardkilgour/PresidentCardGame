#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Online mode for the console UI.

Responsibilities (in order):
  1. Session restore — reuse a saved token if one exists
  2. Login — anonymous or with credentials
  3. Lobby — list/create/join games
  4. Game loop — wait for turns, display state, send moves

Display logic (translating server state to console output) lives in display.py.
"""
from __future__ import annotations

import threading
from pathlib import Path

import yaml

from president.ui.Console.display import meld_from_server, show_and_play

_SESSION_FILE = Path(__file__).parent / '.session.json'
_CONFIG_PATH  = Path(__file__).parents[3] / 'config' / 'config.yaml'


def run_online(server_addr: str, username: str | None, password: str | None,
               player_name: str | None = None) -> None:
    from president.client.server_client import ServerClient

    url       = _normalise_url(server_addr)
    rejoin_ev = threading.Event()
    rejoined  = False

    # --- Try to restore saved session ---
    client = ServerClient.restore_session(str(_SESSION_FILE), server_addr)
    if client:
        print('Attempting to reconnect with saved session…')
        client.on_rejoin_game = rejoin_ev.set
        try:
            client.connect()
            client.save_session(str(_SESSION_FILE))
            rejoin_ev.wait(timeout=2.0)
            if rejoin_ev.is_set():
                print(f'Reconnected as {client.username}')
                rejoined = True
            else:
                print('Session expired — logging in fresh.')
                client.disconnect()
                client = None
        except Exception as e:
            print(f'Reconnect failed ({e}) — logging in fresh.')
            client = None

    # --- Fresh login ---
    if client is None:
        client = ServerClient(url)
        client.on_rejoin_game = rejoin_ev.set
        if username and password:
            print(f'Logging in as {username}…')
            if not client.login(username, password):
                print('Login failed — check credentials.')
                return
        else:
            if not player_name:
                try:
                    cfg = yaml.safe_load(_CONFIG_PATH.read_text())
                    player_name = next(
                        (cfg[k]['name'] for k in ['player1', 'player2', 'player3', 'player4']
                         if cfg[k].get('console')),
                        None,
                    )
                except Exception:
                    pass
            anon = client.login_anonymous(name=player_name)
            print(f'Joined anonymously as: {anon}')
        try:
            client.connect()
            client.save_session(str(_SESSION_FILE))
        except Exception as e:
            print(f'Could not connect to {url}: {e}')
            return
        print(f'Connected to {url}')
        rejoin_ev.wait(timeout=2.0)
        if rejoin_ev.is_set():
            print('Rejoined active game.')
            rejoined = True

    # --- Lobby (skipped if rejoined) ---
    if not rejoined:
        _lobby(client)

    # --- Game loop ---
    _game_loop(client)

    client.disconnect()


# ---------------------------------------------------------------------------
# Lobby
# ---------------------------------------------------------------------------

def _lobby(client) -> None:
    def _print(games):
        print(f'\n{"─" * 52}')
        print('  Ongoing games:')
        if games:
            for i, g in enumerate(games):
                print(f'    [{i}] {g["id"][:8]}…  ({", ".join(g["players"])})')
        else:
            print('    (none)')
        print('    [N] Create a new game')
        print('    [R] Refresh game list')
        print(f'{"─" * 52}')

    games = client.list_games()
    _print(games)

    while True:
        try:
            choice = input('  Pick a game, N or R: ').strip().upper()
        except (EOFError, KeyboardInterrupt):
            client.disconnect()
            return

        if choice == 'R':
            games = client.list_games()
            _print(games)
            continue

        if choice == 'N':
            _create_and_configure_game(client)
            return

        try:
            idx = int(choice)
            if 0 <= idx < len(games):
                client.join_game(games[idx]['id'])
                print(f'Joined game {games[idx]["id"][:8]}…')
                print('Waiting for the game to start…')
                return
        except ValueError:
            pass
        print('  Invalid choice.')


def _create_and_configure_game(client) -> None:
    """Create a game, optionally fill seats with AI, then start."""
    game_id = client.create_game()
    if not game_id:
        print('Failed to create game.')
        return
    print(f'\nGame created: {game_id[:8]}…')

    ai_difficulties = ['Easy', 'Medium', 'Hard']
    print('\nAdd AI opponents (seats 1–3). Press Enter to skip a seat.')
    for seat in range(1, 4):
        try:
            raw = input(
                f'  Seat {seat} difficulty [Easy/Medium/Hard, Enter=Medium, S=skip]: '
            ).strip()
        except (EOFError, KeyboardInterrupt):
            break

        if raw.upper() == 'S':
            continue
        diff = raw.capitalize() if raw.capitalize() in ai_difficulties else 'Medium'
        ai_name = f'CPU-{seat}'
        client.add_ai_player(seat, ai_name, diff)
        print(f'  Added {ai_name} ({diff})')

    try:
        input('\nPress Enter to start the game… ')
    except (EOFError, KeyboardInterrupt):
        pass

    print('Starting…')
    if not client.start_game(timeout=10.0):
        print('Start timed out — the server may have started anyway.')


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def _game_loop(client) -> None:
    turn_event   = threading.Event()
    game_over    = threading.Event()
    state_holder: list[dict | None] = [None]

    def on_state(s):
        state_holder[0] = s
        if s.get('is_my_turn'):
            turn_event.set()
        positions = s.get('player_positions', [])
        if sum(1 for p in positions if p != -1) >= 4:
            game_over.set()

    def on_card_played(player_id, cards):
        if isinstance(cards, list) and cards:
            print(f'  {player_id} plays {meld_from_server(cards)}')
        else:
            print(f'  {player_id} passes.')

    def on_hand_won(winner):
        print(f'\n{"─" * 40}')
        print(f'  {winner} wins the hand.')
        print(f'{"─" * 40}')

    def on_hand_started():
        print(f'\n{"═" * 40}')
        print('  New hand starting.')
        print(f'{"═" * 40}')

    replace_queue: list[str] = []   # players awaiting AI replacement

    def on_player_disconnected(username):
        print(f'\n  *** {username} disconnected.')

    def on_replace_available(username):
        if username not in replace_queue:
            replace_queue.append(username)
        print(f'\n  *** {username} timed out — '
              f'type "r {username}" on your turn to replace with AI.')

    def on_player_replaced(username):
        if username in replace_queue:
            replace_queue.remove(username)
        print(f'\n  *** {username} was replaced by AI.')

    def on_player_quit(username):
        if username in replace_queue:
            replace_queue.remove(username)
        print(f'\n  *** {username} left the game.')

    client.on_game_state          = on_state
    client.on_card_played         = on_card_played
    client.on_hand_won            = on_hand_won
    client.on_hand_started        = on_hand_started
    client.on_player_disconnected = on_player_disconnected
    client.on_replace_available   = on_replace_available
    client.on_player_replaced     = on_player_replaced
    client.on_player_quit         = on_player_quit

    client.request_state(timeout=3.0)
    print('\nGame running! Waiting for your turn…  (Ctrl+C to quit)\n')

    while not game_over.is_set():
        turn_event.wait(timeout=30)
        if not turn_event.is_set():
            continue
        turn_event.clear()

        state = client.request_state(timeout=5.0)
        if not state or not state.get('is_my_turn'):
            continue

        show_and_play(client, state, replace_available=replace_queue)

    print('\nGame over!')
    final = state_holder[0]
    if final:
        ranking_names = ['President', 'Vice-President', 'Citizen', 'Scumbag']
        order = sorted(
            [(pos, name)
             for name, pos in zip(final.get('player_names', []),
                                  final.get('player_positions', []))
             if pos != -1],
            key=lambda x: x[0],
        )
        print('  Final standings:')
        for pos, name in order:
            print(f'    {ranking_names[pos]:16} {name}')


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _normalise_url(addr: str) -> str:
    return addr if '://' in addr else f'http://{addr}'
