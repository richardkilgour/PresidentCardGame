#!/usr/bin/env python
"""
Shared client for connecting to a President Card Game server (app.py).

Handles HTTP session login (or anonymous login) and SocketIO game events.
Both PlayLocally.py and the PyGame UI use this to participate in server-hosted games.
"""
from __future__ import annotations

import json
import threading

import requests
import socketio


class ServerClient:
    """
    Thin wrapper over python-socketio that provides blocking helpers for
    common game operations (list, create, join, play) and delivers server
    events via optional callbacks.

    Typical usage:
        client = ServerClient("http://localhost:5000")
        client.login("alice", "secret")   # or client.login_anonymous()
        client.connect()
        games = client.list_games()
        client.join_game(games[0]["id"])
        client.on_game_state = lambda s: print(s)
        ...
        client.play_cards(["4_0", "4_1"])
        client.disconnect()
    """

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
        self.username: str | None = None
        self.game_id: str | None = None
        self.game_state: dict | None = None
        self.games: list = []

        self._http = requests.Session()
        self.sio = socketio.Client(logger=False, engineio_logger=False)

        self._ev_games   = threading.Event()
        self._ev_state   = threading.Event()
        self._ev_created = threading.Event()
        self._ev_started = threading.Event()
        self._ev_quit    = threading.Event()

        # Optional callbacks — set before connect() so none are missed
        self.on_game_state          = None   # (state: dict) -> None
        self.on_player_turn         = None   # (player_name: str) -> None
        self.on_card_played         = None   # (player_id: str, cards: list) -> None
        self.on_hand_won            = None   # (winner: str) -> None
        self.on_game_started        = None   # () -> None
        self.on_hand_started        = None   # () -> None
        self.on_player_joined       = None   # (name: str) -> None
        self.on_player_quit         = None   # (name: str) -> None
        self.on_player_disconnected = None   # (username: str) -> None
        self.on_replace_available   = None   # (username: str) -> None
        self.on_player_replaced     = None   # (username: str) -> None
        self.on_quit_confirmed      = None   # () -> None
        self.on_error               = None   # (message: str) -> None
        self.on_rejoin_game         = None   # () -> None

        self._register_handlers()

    # -------------------------------------------------------------------------
    # Internal SocketIO event handlers
    # -------------------------------------------------------------------------

    def _register_handlers(self):
        sio = self.sio

        @sio.on('update_game_list')
        def _(data):
            self.games = data.get('games', [])
            self._ev_games.set()

        @sio.on('current_game_state')
        def _(data):
            self.game_state = data
            self._ev_state.set()
            if self.on_game_state:
                self.on_game_state(data)

        @sio.on('notify_player_turn')
        def _(data):
            # Automatically fetch fresh state on every turn notification so
            # all clients (including non-active ones) see the latest board.
            if sio.connected:
                sio.emit('request_game_state', {})
            if self.on_player_turn:
                self.on_player_turn(data.get('player'))

        @sio.on('card_played')
        def _(data):
            if self.on_card_played:
                self.on_card_played(data.get('player_id'), data.get('card_id'))

        @sio.on('hand_won')
        def _(data):
            if self.on_hand_won:
                self.on_hand_won(data.get('winner'))

        @sio.on('notify_game_started')
        def _(data):
            self._ev_started.set()
            if self.on_game_started:
                self.on_game_started()

        @sio.on('notify_hand_start')
        def _(data):
            if self.on_hand_started:
                self.on_hand_started()

        @sio.on('notify_player_joined')
        def _(data):
            if self.on_player_joined:
                self.on_player_joined(data.get('new_player'))

        @sio.on('player_quit')
        def _(data):
            if self.on_player_quit:
                self.on_player_quit(data.get('username'))

        @sio.on('player_disconnected')
        def _(data):
            if self.on_player_disconnected:
                self.on_player_disconnected(data.get('username', ''))

        @sio.on('replace_available')
        def _(data):
            if self.on_replace_available:
                self.on_replace_available(data.get('username', ''))

        @sio.on('player_replaced')
        def _(data):
            if self.on_player_replaced:
                self.on_player_replaced(data.get('username', ''))

        @sio.on('quit_confirmed')
        def _(data):
            self._ev_quit.set()
            if self.on_quit_confirmed:
                self.on_quit_confirmed()

        @sio.on('game_created')
        def _(data):
            self.game_id = data.get('game_id')
            self._ev_created.set()

        @sio.on('error')
        def _(data):
            if self.on_error:
                self.on_error(data.get('message', 'Unknown error'))

        @sio.on('rejoin_game')
        def _(data):
            if self.on_rejoin_game:
                self.on_rejoin_game()

    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------

    def login(self, username: str, password: str) -> bool:
        """Login with existing credentials. Returns True on success."""
        resp = self._http.post(
            f'{self.server_url}/login',
            json={'username': username, 'password': password},
        )
        if resp.ok and resp.json().get('success'):
            self.username = username
            return True
        return False

    def login_anonymous(self, name: str | None = None) -> str:
        """Create a temporary anonymous session. Returns the generated username."""
        payload = {'name': name} if name else {}
        resp = self._http.post(f'{self.server_url}/api/anonymous_login', json=payload)
        resp.raise_for_status()
        self.username = resp.json()['username']
        return self.username

    # -------------------------------------------------------------------------
    # Connection
    # -------------------------------------------------------------------------

    def connect(self, transports=None):
        """Connect the SocketIO client, forwarding the HTTP session cookie.

        Args:
            transports: Optional list of transports to use, e.g. ['websocket'].
                        Defaults to the python-socketio default (polling then upgrade).
        """
        cookie_str = '; '.join(f'{k}={v}' for k, v in self._http.cookies.items())
        kwargs = {'headers': {'Cookie': cookie_str}}
        if transports is not None:
            kwargs['transports'] = transports
        self.sio.connect(self.server_url, **kwargs)

    def disconnect(self):
        if self.sio.connected:
            self.sio.disconnect()

    @property
    def connected(self) -> bool:
        return self.sio.connected

    # -------------------------------------------------------------------------
    # Game management
    # -------------------------------------------------------------------------

    def list_games(self, timeout: float = 5.0) -> list:
        """Fetch the current game list from the server (blocking)."""
        self._ev_games.clear()
        self.sio.emit('refresh_games', {})
        self._ev_games.wait(timeout=timeout)
        return self.games

    def create_game(self, timeout: float = 5.0) -> str | None:
        """Ask the server to create a new game; returns its game_id (blocking)."""
        self._ev_created.clear()
        self.sio.emit('new_game', {})
        self._ev_created.wait(timeout=timeout)
        return self.game_id

    def join_game(self, game_id: str):
        """Join an existing game (non-blocking; state arrives via on_game_state)."""
        self.game_id = game_id
        self.sio.emit('join_game', {'game_id': game_id})

    def add_ai_player(self, seat: int, name: str, difficulty: str = 'Medium'):
        """Add an AI player to a game we own (before start).
        difficulty: 'Easy' | 'Medium' | 'Hard'
        """
        self.sio.emit('add_ai_player', {
            'opponentIndex': seat,
            'aiName': name,
            'aiDifficulty': difficulty,
        })

    def start_game(self, timeout: float = 15.0) -> bool:
        """Start the game we own; blocks until notify_game_started or timeout."""
        self._ev_started.clear()
        self.sio.emit('start_game', {})
        return self._ev_started.wait(timeout=timeout)

    def quit_game(self, timeout: float = 5.0) -> bool:
        """Leave the current game. The server replaces us with AI (slot stays reserved).
        Blocks until quit_confirmed or timeout; returns True on clean confirmation."""
        self._ev_quit.clear()
        self.sio.emit('quit_game', {})
        return self._ev_quit.wait(timeout=timeout)

    def replace_with_ai(self, username: str) -> None:
        """Replace a disconnected player with an AI. Any player in the game may call this
        once the server has emitted replace_available for that username."""
        self.sio.emit('replace_with_ai', {'username': username})

    # -------------------------------------------------------------------------
    # Gameplay
    # -------------------------------------------------------------------------

    def play_cards(self, cards):
        """Send a play.
        cards: 'PASSED'  OR  list of 'value_suit' strings, e.g. ['4_0', '4_1']
        """
        self.sio.emit('play_cards', {'cards': cards})

    def request_state(self, timeout: float = 5.0) -> dict | None:
        """Explicitly request current game state (blocking)."""
        self._ev_state.clear()
        self.sio.emit('request_game_state', {})
        self._ev_state.wait(timeout=timeout)
        return self.game_state

    def logout(self) -> None:
        """Destroy the server-side HTTP session."""
        try:
            self._http.post(f'{self.server_url}/logout')
        except Exception:
            pass
        self.username = None

    # -------------------------------------------------------------------------
    # Session persistence (for reconnect)
    # -------------------------------------------------------------------------

    def save_session(self, path: str) -> None:
        data = {'server_url': self.server_url, 'username': self.username,
                'cookies': dict(self._http.cookies)}
        try:
            with open(path, 'w') as f:
                json.dump(data, f)
        except OSError:
            pass

    @classmethod
    def restore_session(cls, path: str, server_url: str | None = None) -> 'ServerClient | None':
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        saved_url = data.get('server_url', '').rstrip('/')
        if server_url:
            norm = (server_url if '://' in server_url else f'http://{server_url}').rstrip('/')
            if norm != saved_url:
                return None
        client = cls(saved_url)
        for name, value in data.get('cookies', {}).items():
            client._http.cookies.set(name, value)
        client.username = data.get('username')
        return client
