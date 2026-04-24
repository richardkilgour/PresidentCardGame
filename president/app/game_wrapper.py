from president.core.Episode import State
from president.core.GameMaster import GameMaster
from president.core.Meld import Meld
from president.core.PlayingCard import PlayingCard


class GameWrapper(GameMaster):
    def __init__(self, game_id, listener):
        super().__init__()
        self.game_id = game_id
        self.add_listener(listener)
        self.high_score = 0
        self.low_score = 0

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
