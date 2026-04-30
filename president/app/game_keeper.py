from __future__ import annotations

from president.players.AsyncPlayer import AsyncPlayer


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
        """Return a list of all the games the given player is in as a live human."""
        game_list = []
        for gid, game in self._games.items():
            if any(isinstance(p, AsyncPlayer) and p.name == player_id
                   for p in game.player_manager.players if p):
                game_list.append(gid)
        return game_list

    def find_owners_game(self, user_id):
        # Find the game where the user is the owner
        return next((gid for gid, game in self._games.items() if game.player_manager.players[0].name == user_id), None)

    def game_list(self):
        result = []
        for game_id, gm in self._games.items():
            seats = []
            for player in gm.player_manager.players:
                if player is None:
                    seats.append(None)
                elif isinstance(player, AsyncPlayer):
                    seats.append({"type": "human", "name": player.name})
                else:
                    seats.append({"type": "ai", "name": player.name})
            result.append({
                "id": game_id,
                "status": "live" if gm.episode else "waiting",
                "seats": seats,
                "players": [p.name for p in gm.player_manager.players if p],
            })
        return result

    def get_player_names(self, game_id):
        if game_id in self._games:
            return [player.name if player else None for player in self._games[game_id].player_manager.players]
        return []

    def find_reserved_game(self, user_id: str) -> str | None:
        """Return the game_id where this user has a reserved (AI-held) slot."""
        for gid, game in self._games.items():
            if user_id in game.reserved_slots.values():
                return gid
        return None
