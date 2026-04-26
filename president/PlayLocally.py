#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Play a game of PresidentCardGame locally or connect to a running server.

Usage:
    python PlayLocally.py                                   # offline, new game
    python PlayLocally.py --restore crash.json              # resume offline game
    python PlayLocally.py --display crash.json              # display checkpoint, no run
    python PlayLocally.py --server localhost:5000           # join server (anonymous)
    python PlayLocally.py --server 192.168.1.5:5000 \\
                          --username alice --password s3cr3t # join as registered user
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading

import yaml

from president.core.GameRecord import GameRecord
from president.core.GameMaster import GameMaster
from president.core.PlayerRegistry import PlayerRegistry
from president.core.PlayingCard import PlayingCard
from president.players.ConsolePlayer import ConsolePlayer

_SESSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.session.json')

# ---------------------------------------------------------------------------
# Offline entry point (unchanged from original)
# ---------------------------------------------------------------------------

def run_offline(args):
    logging.basicConfig(
        handlers=[logging.FileHandler('test.log', 'w', 'utf-8')],
        level=logging.NOTSET
    )

    config = yaml.safe_load(open("config/config.yaml"))
    registry = PlayerRegistry.from_config(config)
    gm = GameMaster(registry=registry)

    if args.restore:
        print(f"Restoring from {args.restore}...")
        print(GameRecord.display(args.restore))
        record = GameRecord.restore(args.restore, gm, registry)
        print("Restore complete — resuming game.")
    else:
        record = GameRecord(gm)
        for key in ['player1', 'player2', 'player3', 'player4']:
            p = config[key]
            if p.get('console', False):
                gm.add_player(ConsolePlayer(p['name']))
            else:
                gm.make_player(p['type'], p['name'])
        gm.start(number_of_rounds=1000)

    gm.add_listener(record)
    gm.set_record(record)

    for player in gm.player_manager.players:
        if isinstance(player, ConsolePlayer):
            player.set_record(record)

    done = False
    while not done:
        done = gm.step()

    if done:
        record.mark_complete()
        print("Game complete.")
        print(gm.position_stats_str())


# ---------------------------------------------------------------------------
# Online game loop helpers — use ConsolePlayer display for identical UX
# ---------------------------------------------------------------------------

def _normalise_url(addr: str) -> str:
    return addr if '://' in addr else f'http://{addr}'


def _meld_from_server(meld_cards: list):
    """Convert [[value, suit_str], ...] from the server into a Meld object."""
    from president.core.Meld import Meld
    meld = Meld()
    for v, s in meld_cards:
        card = PlayingCard(v * 4 + PlayingCard.suit_list.index(s))
        meld = Meld(card, meld)
    return meld


def _status_from_server(raw):
    """Convert a server player_status value into a Meld/None that ConsolePlayer understands."""
    from president.core.Meld import Meld
    if raw == 'Waiting' or raw == '␆':
        return None
    if raw == 'Passed':
        return Meld()
    if isinstance(raw, list):
        return _meld_from_server(raw)
    return str(raw)   # position string like "President"


class _PlayerProxy:
    """Minimal player-like object used to populate ConsolePlayer.players."""
    def __init__(self, name: str, card_count: int):
        self.name = name
        self._card_count = card_count

    def report_remaining_cards(self) -> int:
        return self._card_count


def _console_for_state(state: dict) -> ConsolePlayer:
    """Build a ConsolePlayer populated with the current server game state."""
    console = ConsolePlayer(state['player_names'][0])

    # Reconstruct hand so __str__ and possible_plays work correctly
    console._hand = [
        PlayingCard(v * 4 + PlayingCard.suit_list.index(s))
        for v, s, _ in state['player_hand']
    ]

    opp_names  = state['player_names'][1:]
    opp_counts = state['opponent_cards']
    console.players = [console] + [
        _PlayerProxy(n, c) for n, c in zip(opp_names, opp_counts)
    ]
    console.player_status = [_status_from_server(s) for s in state['player_status']]
    return console


def _show_and_play(client, state: dict):
    """Display the table via ConsolePlayer and send the chosen play to the server."""
    from president.core.Meld import Meld

    console = _console_for_state(state)
    console._show_table()

    # Build Meld options from server-computed possible_melds
    options: list[Meld] = [_meld_from_server(m) for m in state.get('possible_melds', [])]
    if state.get('can_pass'):
        options.append(Meld())

    print("Select a play:")
    for i, meld in enumerate(options):
        split_str = ''
        if meld.cards and console.will_split(meld):
            split_str = '  (split)'
        print(f'  [{i}] {meld}{split_str}')
    print("  [q] Save and quit")

    while True:
        try:
            raw = input("\nYour choice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print('\nDisconnecting...')
            client.disconnect()
            sys.exit(0)

        if raw in ('q', 'quit'):
            client.disconnect()
            sys.exit(0)

        if not raw:
            selected = options[0]
        else:
            try:
                idx = int(raw)
                if 0 <= idx < len(options):
                    selected = options[idx]
                else:
                    print(f'  Please enter a number from 0 to {len(options) - 1}.')
                    continue
            except ValueError:
                print(f'  Please enter a number from 0 to {len(options) - 1}, or q to quit.')
                continue

        if not selected.cards:
            client.play_cards('PASSED')
        else:
            cards = [f"{c.get_value()}_{c.get_suit()}" for c in selected.cards]
            client.play_cards(cards)
        return


# ---------------------------------------------------------------------------
# Online mode: connect, lobby, game loop
# ---------------------------------------------------------------------------

def run_online(server_addr: str, username: str | None, password: str | None,
               player_name: str | None = None):
    from president.client.server_client import ServerClient

    url = _normalise_url(server_addr)
    rejoin_ev = threading.Event()
    rejoined = False

    # --- Try to restore saved session ---
    client = ServerClient.restore_session(_SESSION_FILE, server_addr)
    if client:
        print('Attempting to reconnect with saved session…')
        client.on_rejoin_game = rejoin_ev.set
        try:
            client.connect()
            client.save_session(_SESSION_FILE)
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

    # --- Fresh login if restore failed or no saved session ---
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
                    cfg = yaml.safe_load(open("config/config.yaml"))
                    player_name = next(
                        (cfg[k]['name'] for k in ['player1', 'player2', 'player3', 'player4']
                         if cfg[k].get('console')),
                        None
                    )
                except Exception:
                    pass
            anon = client.login_anonymous(name=player_name)
            print(f'Joined anonymously as: {anon}')
        try:
            client.connect()
            client.save_session(_SESSION_FILE)
        except Exception as e:
            print(f'Could not connect to {url}: {e}')
            return
        print(f'Connected to {url}')
        rejoin_ev.wait(timeout=2.0)
        if rejoin_ev.is_set():
            print('Rejoined active game.')
            rejoined = True

    # --- Game selection (skipped if rejoined) ---
    if not rejoined:
        def _print_lobby(games):
            print(f'\n{"─" * 52}')
            print('  Ongoing games:')
            if games:
                for i, g in enumerate(games):
                    players_str = ', '.join(g['players'])
                    print(f'    [{i}] {g["id"][:8]}…  ({players_str})')
            else:
                print('    (none)')
            print('    [N] Create a new game')
            print('    [R] Refresh game list')
            print(f'{"─" * 52}')

        games = client.list_games()
        _print_lobby(games)

        while True:
            try:
                choice = input('  Pick a game, N or R: ').strip().upper()
            except (EOFError, KeyboardInterrupt):
                client.disconnect()
                return

            if choice == 'R':
                games = client.list_games()
                _print_lobby(games)
                continue

            if choice == 'N':
                _create_and_configure_game(client)
                break
            try:
                idx = int(choice)
                if 0 <= idx < len(games):
                    client.join_game(games[idx]['id'])
                    print(f'Joined game {games[idx]["id"][:8]}…')
                    print('Waiting for the game to start…')
                    break
            except ValueError:
                pass
            print('  Invalid choice.')

    # --- Game loop ---
    turn_event = threading.Event()
    state_holder: list[dict | None] = [None]
    game_over = threading.Event()

    played_out_count = [0]

    def on_state(s):
        state_holder[0] = s
        if s.get('is_my_turn'):
            turn_event.set()

    def on_card_played(player_id, cards):
        if isinstance(cards, list) and cards:
            meld = _meld_from_server(cards)
            print(f'  {player_id} plays {meld}')
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

    def on_played_out_check(s):
        # Count how many players have a position assigned
        if s:
            positions = s.get('player_positions', [])
            finished = sum(1 for p in positions if p != -1)
            if finished >= 4:
                game_over.set()

    client.on_game_state = lambda s: (on_state(s), on_played_out_check(s))
    client.on_card_played = on_card_played
    client.on_hand_won = on_hand_won
    client.on_hand_started = on_hand_started

    # Request initial state now (in case game already started)
    client.request_state(timeout=3.0)

    print('\nGame running! Waiting for your turn…  (Ctrl+C to quit)\n')

    while not game_over.is_set():
        turn_event.wait(timeout=30)
        if not turn_event.is_set():
            continue
        turn_event.clear()

        # Request fresh state to guarantee hand and possible_melds are current
        state = client.request_state(timeout=5.0)
        if not state or not state.get('is_my_turn'):
            continue

        _show_and_play(client, state)

    print('\nGame over!')
    final = state_holder[0]
    if final:
        positions = final.get('player_positions', [])
        names = final.get('player_names', [])
        ranking_names = ['President', 'Vice-President', 'Citizen', 'Scumbag']
        order = sorted(
            [(pos, name) for name, pos in zip(names, positions) if pos != -1],
            key=lambda x: x[0]
        )
        print('  Final standings:')
        for pos, name in order:
            print(f'    {ranking_names[pos]:16} {name}')

    client.disconnect()


def _create_and_configure_game(client):
    """Create a game, optionally add AI players, then start."""
    game_id = client.create_game()
    if not game_id:
        print('Failed to create game.')
        return
    print(f'\nGame created: {game_id[:8]}…')

    ai_difficulties = ['Easy', 'Medium', 'Hard']
    seats_filled = 1  # we occupy seat 0

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
        seats_filled += 1

    try:
        input('\nPress Enter to start the game… ')
    except (EOFError, KeyboardInterrupt):
        pass

    print('Starting…')
    if not client.start_game(timeout=10.0):
        print('Start timed out — the server may have started anyway.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Play PresidentCardGame locally or online.')
    parser.add_argument('--restore', type=str, metavar='FILE',
                        help='Resume a game from a checkpoint file (offline only).')
    parser.add_argument('--display', type=str, metavar='FILE',
                        help='Display a checkpoint file and exit (offline only).')
    parser.add_argument('--server', type=str, metavar='HOST[:PORT]',
                        help='Server address for online play (e.g. localhost:5000).')
    parser.add_argument('--username', type=str, help='Username for online play.')
    parser.add_argument('--password', type=str, help='Password for online play.')
    parser.add_argument('--name', type=str, help='Display name for anonymous online play.')
    args = parser.parse_args()

    if args.display:
        print(GameRecord.display(args.display))
        return

    if args.server:
        run_online(args.server, args.username, args.password, args.name)
        return

    run_offline(args)


if __name__ == '__main__':
    main()
