import json
import os
import sched
import threading
import time
import uuid
from datetime import timedelta

from flask import Flask, request, redirect, session, render_template, url_for, flash, jsonify
from flask_socketio import SocketIO, join_room

from werkzeug.security import generate_password_hash, check_password_hash

from asshole.app.game_wrapper import GameWrapper
from asshole.core.AbstractPlayer import possible_plays
from asshole.core.CardGameListener import CardGameListener
from asshole.core.Meld import Meld
from asshole.core.PlayingCard import PlayingCard
from asshole.players.AsyncPlayer import AsyncPlayer
from asshole.players.PlayerHolder import PlayerHolder
from asshole.players.PlayerSimple import PlayerSimple
from asshole.players.PlayerSplitter import PlayerSplitter

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.permanent_session_lifetime = timedelta(days=7)
socketio = SocketIO(app, cors_allowed_origins="*", manage_session=True)

PLAYER_DB = "db/players.json"
GAME_DB = "db/games.json"

def load_data(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_data(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


# Dictionary of user IDs and passwords
players = load_data(PLAYER_DB)

# Singleton
class Games:
    _instance = None
    _games = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Games, cls).__new__(cls)
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


def step_games(sc):
    games = Games().get_games()
    for game_id, game_wrapper in games.items():
        if game_wrapper.episode:
            game_wrapper.step()
    # Schedule the function to run again after 1 second
    sc.enter(1, 1, step_games, (sc,))

scheduler = sched.scheduler(time.time, time.sleep)

def run_scheduler():
    scheduler.enter(1, 1, step_games, (scheduler,))
    scheduler.run()

# Start the scheduler in a separate thread
scheduler_thread = threading.Thread(target=run_scheduler)
scheduler_thread.start()

def cards_to_list(cards):
    return [[card.get_value(), card.suit_str()] for card in cards]

class EventBroadcaster(CardGameListener):
    def __init__(self, game_id):
        super().__init__()
        self.game_id = game_id

    def notify_player_joined(self, new_player, position):
        print(f"notify_player_joined called from listener")
        socketio.emit("notify_player_joined", {"game_id": self.game_id, "new_player": new_player.name, "position": position})#, to=self.game_id)

    def notify_game_stated(self):
        socketio.emit('notify_game_stated', {'game_id': self.game_id})

    def notify_hand_start(self, starter):
        print(f"notify_hand_start called from listener")
        socketio.emit('notify_hand_start',{"starter": starter.name})#, to=self.game_id)

    def notify_hand_won(self, winner):
        print(f"notify_hand_won called from listener")
        socketio.emit('hand_won', {"winner": winner.name})#, to=self.game_id)

    def notify_played_out(self, player, pos):
        # Someone just lost all their cards
        print(f"notify_played_out called from listener")
        socketio.emit('notify_played_out', {"player": player.name, "pos": pos})#, to=self.game_id)

    def notify_play(self, player, meld):
        print(f"notify_play called from listener")
        socketio.emit('card_played', {
            'player_id': player.name,
            'card_id': cards_to_list(meld.cards),
        })#, to=self.game_id)


@socketio.on('connect')
def handle_connect():
    # Associate Flask session data with this socket if needed
    session['socket_id'] = request.sid
    session.modified = True  # Important: Mark the session as modified


@socketio.on('disconnect')
def handle_disconnect():
    session['socket_id'] = None
    session.modified = True  # Important: Mark the session as modified


@app.route('/')
def index():
    # TODO: If this user is currently in a game, send them there
    return render_template(
        'home.html',
        games=Games().get_games(),
        username=session.get('user')
    )


@app.route('/register_user', methods=['POST'])
def register():
    data = request.get_json()
    username, password = data.get('username'), data.get('password')

    if username in players:
        flash('Username already exists')
        return redirect(url_for('register'))

    # Assign a unique ID to the new user
    players[username] = generate_password_hash(password)
    session['user'] = username
    session['logged_in'] = True
    save_data(PLAYER_DB, players)
    return redirect(url_for('register'))


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username, password = data.get('username'), data.get('password')

    if username in players and check_password_hash(players[username], password):
        session['user'] = username
        session['logged_in'] = True
        return jsonify({"success": True})

    flash('Invalid username or password')
    return jsonify({"success": False, "error": "Invalid credentials"})


@socketio.on('logout')
def logout():
    user = session.get('user')
    # TODO: leave_room(game_id)
    session['logged_in'] = False
    if user:
        session.pop('user')
    session.modified = True  # Important: Mark the session as modified


# Also allow log out via HTTP
@app.route('/logout', methods=['POST'])
def http_logout():
    session['logged_in'] = False
    # TODO: leave_room(game_id)
    if 'user' in session:
        session.pop('user')
    return redirect(url_for('index'))


@socketio.on('refresh_games')
def send_game_list():
    """Send game list to all clients"""
    games = Games().get_games()
    game_list = [{"id": game_id, "players": [player.name for player in gm.players if player]} for game_id, gm in games.items()]
    socketio.emit('update_game_list', {"games": game_list})


@app.route('/get_user_info')
def get_user_info():
    if 'logged_in' in session and session['logged_in']:
        info = {"logged_in": True, "username": session.get('user')}
    else:
        info = {"logged_in": False}
    return jsonify(info)


def add_human_player(user_id, game_id):
    # Add 'human' player to the game
    player = AsyncPlayer(user_id)
    Games().get_game(game_id).add_player(player) # Should trigger a callback to EventBroadcaster
    print(f"User {user_id} joined game {game_id}")


@socketio.on('new_game')
def create_game():
    user = session.get('user')  # Retrieve user from session

    if not user:
        print("Unauthorized game creation attempt.")
        socketio.emit('error', {'message': 'You must be logged in to start a game'})
        return

    game_id = str(uuid.uuid4())
    Games().add_game(game_id, GameWrapper(game_id, EventBroadcaster(game_id)))
    # Notify the user and update the game list for all players
    socketio.emit('game_created', {'game_id': game_id})
    print(f"New game created by {user}: {game_id}")
    join_room(game_id)
    add_human_player(user, game_id)


@socketio.on('join_game')
def handle_join_game(data):
    user_id = session.get('user')

    if not user_id:
        socketio.emit('error', {'message': 'You must be logged in to join a game'})
        return

    game_id = data.get("game_id")
    join_room(game_id)
    add_human_player(user_id, game_id)


def find_owners_game(user_id):
    # Find the game where the user is the owner
    return next((gid for gid, game in Games().get_games().items() if game.players[0].name == user_id), None)


@socketio.on('add_ai_player')
def add_ai_player(data):
    user_id = session.get('user')
    game_id = find_owners_game(user_id)

    if not game_id or user_id != Games().get_game(game_id).players[0].name:
        socketio.emit('error', {'message': 'Only the game owner can add AI players'})
        return

    opponent_index = data["opponentIndex"]
    ai_name = data["aiName"]
    ai_difficulty = data["aiDifficulty"]
    if ai_difficulty == "Easy":
        new_ai = PlayerSimple(ai_name)
    elif ai_difficulty == "Medium":
        new_ai = PlayerHolder(ai_name)
    else:
        new_ai = PlayerSplitter(ai_name)

    # Add AI to the correct position
    Games().get_game(game_id).add_player(new_ai, opponent_index)
    send_game_state()


@socketio.on('view_game')
def view_game():
    # TODO: view game but can't interact with it (unless someone leaves)
    pass


# Add new socket event handler
@socketio.on('start_game')
def start_game():
    player_id = session.get('user')
    game_id = find_owners_game(player_id)

    if not game_id or game_id not in Games().get_games():
        return {'error': 'Game not found'}

    Games().get_game(game_id).start()


def get_player_names(game_id):
    if game_id in Games().get_games():
        return [player.name if player else None for player in Games().get_game(game_id).players]
    return []

def find_valid_game(user_id, game_id = None):
    games = Games().get_games()

    # Check if the provided game_id is valid
    if not game_id or game_id not in games:
        # Search for a valid game based on user_id
        game_id = next((gid for gid, gm in games.items() if user_id in get_player_names(gid)), None)

    # Return None if no valid game_id is found
    if not game_id or game_id not in games:
        return None

    return game_id



def get_game_state(user_id, game_id=None):
    """Find the game for a user and return structured game state data."""
    if not user_id:
        return None  # No valid session

    game_id = find_valid_game(user_id, game_id)

    if not game_id:
        return None # Invalid game ID or player not in game

    gm = Games().get_game(game_id)
    player_names = get_player_names(game_id)

    if user_id not in player_names:
        raise KeyError # The check above should ensure this does not happen

    player_index = player_names.index(user_id)
    player = gm.players[player_index]

    # Get opponent details - start at (player_index+1) to ensure the correct order
    opponent_details = []
    for i in range(1, 4):
        opponent_index = (i + player_index) % 4
        if gm.players[opponent_index]:
            opponent_details.append({
                "name": gm.players[opponent_index].name,
                "card_count": gm.players[opponent_index].report_remaining_cards(),
                "status": "Waiting",  # TODO: Replace with actual status, but currently unused
            })
        else:
            opponent_details.append({"name": None, "card_count": 0, "status": "Absent"})

    playable_cards = [c.cards[-1] for c in possible_plays(player._hand, player.target_meld)[:-1]]
    player_cards = []
    for card in player._hand:
        # Check each card to see if it's playable
        playable = card in playable_cards
        player_cards.append([card.get_value(), card.suit_str(), playable])


    return {
        "game_id": game_id,
        "player_id": user_id,
        "opponent_details": opponent_details,
        "is_owner": (gm.players[0].name == user_id),
        "player_hand": player_cards
    }


@app.route('/game/<game_id>')
def show_game(game_id):
    if not session.get('logged_in', False):
        # TODO: allow anonymous players
        return redirect(url_for('index'))

    user = session.get('user')
    game_state = get_game_state(user, game_id)

    if not game_state:
        return "Game not found or unauthorized access", 404

    return render_template('game.html',
                           player_id=game_state["player_id"],
                           game_id=game_state["game_id"],
                           opponent_details=game_state["opponent_details"],
                           is_owner=game_state["is_owner"])


@socketio.on('request_game_state')
def send_game_state():
    """Send the full game state to the requesting player."""
    user_id = session.get('user')
    game_state = get_game_state(user_id)

    if not game_state:
        socketio.emit('error', {'message': 'Game not found or unauthorized access'})
        return

    # Emit the entire game state in one message
    socketio.emit('current_game_state', game_state)


@socketio.on('play_cards')
def handle_play_card(data):
    user_id = session.get('user')
    game_id = find_valid_game(user_id)

    if not game_id or game_id not in Games().get_games():
        return {'error': 'Game not found'}

    game = Games().get_game(game_id)
    if game.episode.active_players[0].name != user_id:
        return {'error': 'Not your turn'}

    # data['cards'] are a string like '5_0'
    meld = Meld()
    if data['cards'] != 'pass':
        for card in data['cards']:
            value, suit = card.split('_')
            meld = Meld(PlayingCard(int(value)*4+int(suit)), meld)

    # Process the play
    for p in game.players:
        if p.name == user_id:
            p.add_play(meld)

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, use_reloader=False)
