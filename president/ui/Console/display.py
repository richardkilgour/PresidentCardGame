#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Console rendering for online game state.

Translates server state dicts into ConsolePlayer display calls and handles the
interactive turn loop (show table → read input → send play).

Nothing in this module knows about sessions, lobbies, or reconnection — it only
knows how to display a game state and send a single move.
"""
from __future__ import annotations

import sys

from president.core.Meld import Meld
from president.core.PlayingCard import PlayingCard
from president.players.ConsolePlayer import ConsolePlayer


class PlayerProxy:
    """Minimal player-like object used to fill ConsolePlayer.players for display."""

    def __init__(self, name: str, card_count: int):
        self.name = name
        self._card_count = card_count

    def report_remaining_cards(self) -> int:
        return self._card_count


def meld_from_server(meld_cards: list) -> Meld:
    """Convert [[value, suit_str], …] from the server into a Meld object."""
    meld = Meld()
    for v, s in meld_cards:
        card = PlayingCard(v * 4 + PlayingCard.suit_list.index(s))
        meld = Meld(card, meld)
    return meld


def status_from_server(raw) -> Meld | None | str:
    """Convert a server player_status value into the form ConsolePlayer expects."""
    if raw in ('Waiting', '␆'):
        return None
    if raw == 'Passed':
        return Meld()
    if isinstance(raw, list):
        return meld_from_server(raw)
    return str(raw)   # position string like "President"


def console_for_state(state: dict) -> ConsolePlayer:
    """Build a display-ready ConsolePlayer from a server game-state dict."""
    console = ConsolePlayer(state['player_names'][0])

    console._hand = [
        PlayingCard(v * 4 + PlayingCard.suit_list.index(s))
        for v, s, _ in state['player_hand']
    ]

    opp_names  = state['player_names'][1:]
    opp_counts = state['opponent_cards']
    console.players = [console] + [
        PlayerProxy(n, c) for n, c in zip(opp_names, opp_counts)
    ]
    console.player_status = [status_from_server(s) for s in state['player_status']]
    return console


def show_and_play(client, state: dict,
                  replace_available: list[str] | None = None) -> None:
    """Display the current table state and send the player's chosen move.

    replace_available — names of disconnected players that can be swapped for AI.
    Any player in the game may replace; the list is maintained by the caller.
    """
    console = console_for_state(state)
    console._show_table()

    options: list[Meld] = [meld_from_server(m) for m in state.get('possible_melds', [])]
    if state.get('can_pass'):
        options.append(Meld())

    print("Select a play:")
    for i, meld in enumerate(options):
        split_str = '  (split)' if meld.cards and console.will_split(meld) else ''
        print(f'  [{i}] {meld}{split_str}')

    if replace_available:
        print()
        for name in replace_available:
            print(f'  [r {name}] Replace {name} with AI')

    print("  [q] Quit game")

    while True:
        try:
            raw = input("\nYour choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print('\nQuitting game (you will be replaced by AI)…')
            client.quit_game(timeout=3.0)
            client.disconnect()
            sys.exit(0)

        if raw.lower() in ('q', 'quit'):
            print('Quitting game (you will be replaced by AI)…')
            client.quit_game(timeout=3.0)
            client.disconnect()
            sys.exit(0)

        # Replace command: "r Alice"  (case-insensitive match)
        if replace_available and raw.lower().startswith('r '):
            target_input = raw[2:].strip()
            match = next(
                (n for n in replace_available if n.lower() == target_input.lower()),
                None,
            )
            if match:
                client.replace_with_ai(match)
                print(f'  Replacing {match} with AI…')
                replace_available.remove(match)
                continue
            else:
                print(f'  Unknown player "{target_input}". '
                      f'Available: {", ".join(replace_available)}')
                continue

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
                hint = ', r <name>' if replace_available else ''
                print(f'  Please enter a number from 0 to {len(options) - 1}'
                      f'{hint}, or q to quit.')
                continue

        if not selected.cards:
            client.play_cards('PASSED')
        else:
            cards = [f"{c.get_value()}_{c.get_suit()}" for c in selected.cards]
            client.play_cards(cards)
        return
