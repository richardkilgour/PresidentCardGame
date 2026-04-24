import time

from president.core.Episode import State
from president.core.GameMaster import GameMaster
from president.core.Meld import Meld
from president.core.PlayingCard import PlayingCard
from president.players.AsyncPlayer import AsyncPlayer
from president.players.PlayerHolder import PlayerHolder
from president.players.PlayerSimple import PlayerSimple
from president.players.PlayerSplitter import PlayerSplitter


class GameWrapper(GameMaster):
    def __init__(self, game_id, listener):
        super().__init__()
        self.game_id = game_id
        self.add_listener(listener)
        self.high_score = 0
        self.low_score = 0
        # seat_index → username of the disconnected human the AI is holding for
        self.reserved_slots: dict[int, str] = {}
        # username → {"time": float, "timeout": 10|20, "notified": bool}
        self.disconnect_info: dict[str, dict] = {}

    def on_round_completed(self):
        result = super().on_round_completed()
        for player in self.player_manager.players:
            if player:
                score = player.get_score()
                self.high_score = max(self.high_score, score)
                self.low_score = min(self.low_score, score)
        return result

    @property
    def open_card_index(self):
        return self.episode.open_card_index if self.episode else None

    def can_start(self):
        return not self.episode or self.episode.state == State.INITIALISED

    def play(self, user_id, cards_data):
        if not self.episode:
            return 'Game not started'
        if not self.episode.active_players:
            return 'Round not started'
        if self.episode.active_players[0].name != user_id:
            return 'Not your turn'

        meld = Meld()
        if cards_data != 'PASSED':
            for card in cards_data:
                value, suit = card.split('_')
                meld = Meld(PlayingCard(int(value) * 4 + int(suit)), meld)

        for p in self.player_manager.players:
            if p.name == user_id:
                p.add_play(meld)
        self.episode.step()
        return None

    # -------------------------------------------------------------------------
    # Human / AI helpers
    # -------------------------------------------------------------------------

    def human_players(self) -> list[tuple[int, AsyncPlayer]]:
        """Return (seat, player) pairs for every live human (AsyncPlayer) in the game."""
        return [
            (i, p) for i, p in enumerate(self.player_manager.players)
            if p and isinstance(p, AsyncPlayer)
        ]

    def all_human_usernames(self) -> set[str]:
        """All humans: live AsyncPlayers plus those with reserved (AI-held) slots."""
        names = {p.name for _, p in self.human_players()}
        names.update(self.reserved_slots.values())
        return names

    @staticmethod
    def ai_for_score(score: int):
        """Pick AI difficulty to match the departing human's skill level."""
        if score >= 2:
            return PlayerSplitter   # Hard
        elif score >= -1:
            return PlayerHolder     # Medium
        else:
            return PlayerSimple     # Easy

    def replace_human_with_ai(self, username: str, reserved: bool) -> None:
        """
        Swap a human AsyncPlayer for a score-appropriate AI.
        If reserved=True the slot is held for that human to reclaim later.
        """
        for i, p in enumerate(self.player_manager.players):
            if p and isinstance(p, AsyncPlayer) and p.name == username:
                ai_class = self.ai_for_score(p.get_score())
                ai_player = ai_class(username)   # same name — seamless to other players
                self.swap_player(p, ai_player)
                if reserved:
                    self.reserved_slots[i] = username
                else:
                    self.reserved_slots.pop(i, None)
                break

    def restore_human_player(self, username: str) -> bool:
        """
        Swap the reserved AI slot back to an AsyncPlayer for the returning human.
        Returns True if the swap was performed.
        """
        for seat, reserved_user in list(self.reserved_slots.items()):
            if reserved_user == username:
                ai_player = self.player_manager.players[seat]
                if ai_player is None:
                    return False
                human = AsyncPlayer(username)
                self.swap_player(ai_player, human)
                del self.reserved_slots[seat]
                return True
        return False

    def record_disconnect(self, username: str, other_humans_connected: bool) -> None:
        """Record that a human has dropped; choose the appropriate replacement timeout."""
        timeout = 10 if other_humans_connected else 20
        self.disconnect_info[username] = {
            "time": time.time(),
            "timeout": timeout,
            "notified": False,
        }

    def clear_disconnect(self, username: str) -> None:
        self.disconnect_info.pop(username, None)
