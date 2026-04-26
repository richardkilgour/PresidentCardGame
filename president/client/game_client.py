#!/usr/bin/env python
"""
GameClient protocol — the interface every concrete client must satisfy.

Both ServerClient (online, connects to Flask/SocketIO server) and any future
LocalClient (offline, wraps GameMaster directly) implement this contract.
UI modules (Console, PyGame) are typed against GameClient so they don't care
which concrete client they hold.

Usage:
    from president.client.game_client import GameClient
    def run(client: GameClient) -> None: ...
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class GameClient(Protocol):
    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    username: str | None

    # ------------------------------------------------------------------
    # Event callbacks  (all optional — None by default; set before connect)
    # ------------------------------------------------------------------
    on_game_state:          Callable | None   # (state: dict)
    on_player_turn:         Callable | None   # (player_name: str)
    on_card_played:         Callable | None   # (player_id: str, cards: list)
    on_hand_won:            Callable | None   # (winner: str)
    on_game_started:        Callable | None   # ()
    on_hand_started:        Callable | None   # ()
    on_player_joined:       Callable | None   # (name: str)
    on_player_quit:         Callable | None   # (name: str)
    on_player_disconnected: Callable | None   # (username: str)
    on_replace_available:   Callable | None   # (username: str) — AI slot ready
    on_player_replaced:     Callable | None   # (username: str) — AI took the slot
    on_quit_confirmed:      Callable | None   # ()
    on_error:               Callable | None   # (message: str)
    on_rejoin_game:         Callable | None   # ()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    @property
    def connected(self) -> bool: ...

    # ------------------------------------------------------------------
    # Lobby
    # ------------------------------------------------------------------
    def list_games(self, timeout: float = 5.0) -> list: ...
    def create_game(self, timeout: float = 5.0) -> str | None: ...
    def join_game(self, game_id: str) -> None: ...
    def add_ai_player(self, seat: int, name: str, difficulty: str = 'Medium') -> None: ...
    def start_game(self, timeout: float = 15.0) -> bool: ...
    def quit_game(self, timeout: float = 5.0) -> bool: ...

    # ------------------------------------------------------------------
    # Gameplay
    # ------------------------------------------------------------------
    def play_cards(self, cards) -> None: ...
    def request_state(self, timeout: float = 5.0) -> dict | None: ...
    def replace_with_ai(self, username: str) -> None: ...
