from __future__ import annotations

import threading

from president.client.server_client import ServerClient

_SUIT_INDEX = {"♠": 0, "♣": 1, "♦": 2, "♥": 3}


class DummyClient:
    def __init__(self, server_url: str):
        self.client = ServerClient(server_url)
        self._game_over = threading.Event()
        self._played_out_count = 0
        self._lock = threading.Lock()

        self.client.on_game_state = self._on_state

        @self.client.sio.on('notify_played_out')
        def _(data):
            with self._lock:
                self._played_out_count += 1
                if self._played_out_count >= 4:
                    self._game_over.set()

    def setup(self):
        self.client.login_anonymous()
        self.client.connect(transports=['websocket'])

    def teardown(self):
        self.client.disconnect()

    @property
    def username(self):
        return self.client.username

    def wait_for_game_over(self, timeout: float = 30.0) -> bool:
        return self._game_over.wait(timeout)

    def _on_state(self, state: dict):
        if state.get("is_my_turn"):
            if state["possible_melds"]:
                first = state["possible_melds"][0]
                cards = [f"{v}_{_SUIT_INDEX[s]}" for v, s in first]
                self.client.play_cards(cards)
            elif state["can_pass"]:
                self.client.play_cards("PASSED")

    def create_game(self) -> str | None:
        return self.client.create_game()

    def join_game(self, game_id: str):
        self.client.join_game(game_id)

    def start_game(self, timeout: float = 15.0) -> bool:
        return self.client.start_game(timeout)

    def quit_game(self, timeout: float = 5.0) -> bool:
        return self.client.quit_game(timeout)

    def disconnect(self):
        self.client.disconnect()

    def reconnect(self):
        """Reconnect using the same HTTP session — server recognises the same user."""
        self.client.connect(transports=['websocket'])
