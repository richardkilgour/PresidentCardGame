# Wrap a Game Master object is a way that can be used by the server app
from president.core.GameMaster import GameMaster


class GameWrapper(GameMaster):
    def __init__(self, game_id, listener):
        super().__init__()
        self.game_id = game_id
        self.add_listener(listener)
