from president.core.CardGameListener import CardGameListener

def cards_to_list(cards):
    return [[card.get_value(), card.suit_str()] for card in cards]


class GameEventHandler(CardGameListener):
    """
    Sends events to all players in a specific game room.
    Acts as a bridge between the game engine and the socket-based UI.
    """

    def __init__(self, socketio, game_id):
        super().__init__()
        self.socketio = socketio
        self.game_id = str(game_id)  # Ensure game_id is a string for room name

    def notify_player_joined(self, new_player, position=None):
        """Notify all players that a new player has joined the game."""
        print(f"notify_player_joined: {new_player.name} in game {self.game_id}")
        self.socketio.emit("notify_player_joined", {
            "game_id": self.game_id,
            "new_player": new_player.name,
            "position": position
        }, room=self.game_id)

    def notify_game_stated(self):
        """Notify all players that the game has started."""
        print(f"notify_game_started for game {self.game_id}")
        self.socketio.emit('notify_game_stated', {
            'game_id': self.game_id
        }, room=self.game_id)

    def notify_hand_start(self):
        """Notify all players that a new hand has started."""
        print(f"notify_hand_start for game {self.game_id}")
        self.socketio.emit('notify_hand_start', {
            'game_id': self.game_id
        }, room=self.game_id)

    def notify_hand_won(self, winner):
        """Notify all players that a hand has been won."""
        print(f"notify_hand_won: {winner.name} in game {self.game_id}")
        self.socketio.emit('hand_won', {
            "winner": winner.name,
            "game_id": self.game_id
        }, room=self.game_id)

    def notify_played_out(self, player, pos):
        """Notify all players that someone has played all their cards."""
        print(f"notify_played_out: {player.name} in position {pos} in game {self.game_id}")
        self.socketio.emit('notify_played_out', {
            "player": player.name,
            "pos": pos,
            "game_id": self.game_id
        }, room=self.game_id)

    def notify_play(self, player, meld):
        """Notify all players that a card or cards have been played."""
        print(f"notify_play: {player.name} played in game {self.game_id}")
        self.socketio.emit('card_played', {
            'player_id': player.name,
            'card_id': cards_to_list(meld.cards),
            'game_id': self.game_id
        }, room=self.game_id)

    def notify_player_turn(self, player):
        """Notify all players whose turn it is."""
        print(f"notify_player_turn: {player.name}'s turn in game {self.game_id} members {self.socketio.server.manager.rooms.get('/').get(self.game_id, [])}")

        self.socketio.emit('notify_player_turn', {
            "player": player.name,
            "game_id": self.game_id
        }, room=self.game_id)

    def notify_cards_swapped(self, player_good, player_bad, num_cards):
        """Notify all players that cards have been swapped between players."""
        print(f"notify_cards_swapped: {num_cards} cards from {player_bad} to {player_good} in game {self.game_id}")
        self.socketio.emit('notify_cards_swapped', {
            "player_good": player_good.name,
            "player_bad": player_bad.name,
            "num_cards": num_cards,
            "game_id": self.game_id
        }, room=self.game_id)
