class GamesKeeper:
    _instance = None
    _games = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GamesKeeper, cls).__new__(cls)
        return cls._instance

    def get_games(self):
        return self._games

    def get_game(self, game_id):
        return self._games[game_id]

    def add_game(self, game_id, game_wrapper):
        self._games[game_id] = game_wrapper

    def remove_game(self, game_id):
        if game_id in self._games:
            del self._games[game_id]

    def add_player(self, game_id, player, position=None):
        self._games[game_id].add_player(player, position)

    def find_player(self, player_id) -> list[str]:
        """Return a list of all the games the given player is in"""
        game_list = []
        for gid, game in self._games.items():
            if player_id in [p.name for p in game.players if p]:
                game_list.append(gid)
        return game_list

    def find_owners_game(self, user_id):
        # Find the game where the user is the owner
        return next((gid for gid, game in self._games.items() if game.players[0].name == user_id), None)

    def game_list(self):
        return [{"id": game_id, "players": [player.name for player in gm.players if player]} for game_id, gm in
                self._games.items()]

    def get_player_names(self, game_id):
        if game_id in self._games:
            return [player.name if player else None for player in self._games[game_id].players]
        return []
