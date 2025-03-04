import json
import os
import sched
import threading
import time
import uuid
from datetime import timedelta

import bcrypt
from flask import Flask, request, redirect, session, render_template, url_for, flash, jsonify
from flask_socketio import SocketIO, join_room

from werkzeug.security import check_password_hash

from asshole.app.game_event_handler import GameEventHandler
from asshole.app.game_keeper import GamesKeeper
from asshole.app.game_wrapper import GameWrapper
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

# In-memory mapping of user sessions to socket IDs
# In production, consider using Redis or another shared cache
user_socket_map = {}  # {username: {sid1, sid2, ...}}
socket_user_map = {}  # {sid: username}


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

def step_games(sc):
    games = GamesKeeper().get_games()
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


@socketio.on('connect')
def handle_connect(data=None):
    username = session.get("user")
    sid = request.sid

    if username:
        # Associate this socket with the logged-in user
        if username not in user_socket_map:
            user_socket_map[username] = set()
        user_socket_map[username].add(sid)
        socket_user_map[sid] = username

        # Notify user of successful connection
        socketio.emit('connection_status', {'status': 'connected', 'username': username})
        print(f"User {username} connected with socket ID: {sid}")

        # If the user is in a game, add them to the room
        for game_id in GamesKeeper().find_player(username):
            for sid in user_socket_map[username]:
                join_room(game_id, sid)
    else:
        # Allow connection but mark as unauthenticated
        # This gives you a chance to handle login via socket events if needed
        socketio.emit('connection_status', {'status': 'anonymous'})
        print(f"Anonymous socket connection: {sid}")


@socketio.on('disconnect')
def handle_disconnect(data=None):
    sid = request.sid
    username = socket_user_map.get(sid)

    if username:
        # Remove this socket ID from the user's set of active connections
        if username in user_socket_map:
            user_socket_map[username].discard(sid)
            # If user has no active connections left, clean up
            if not user_socket_map[username]:
                del user_socket_map[username]

        # Remove from socket to user mapping
        del socket_user_map[sid]
        print(f"User {username} disconnected socket {sid}")
    else:
        print(f"Anonymous socket disconnected: {sid}")


@socketio.on('authenticate')
def handle_socket_authentication(data):
    """Handle authentication via SocketIO instead of HTTP"""
    sid = request.sid
    username = data.get('username')
    password = data.get('password')

    players = load_data(PLAYER_DB)

    if username in players and bcrypt.checkpw(password.encode(), players[username]["password"].encode()):
        # Set Flask session
        session["user"] = username
        session.modified = True

        # Update socket mappings
        if username not in user_socket_map:
            user_socket_map[username] = set()
        user_socket_map[username].add(sid)
        socket_user_map[sid] = username

        socketio.emit('auth_response', {'success': True, 'username': username})
        print(f"Socket authentication successful for {username}")

        # Check if user is in a game and send relevant game data
        games = load_data(GAME_DB)
        for game_id, game in games.items():
            if username in game["players"]:
                socketio.emit('active_game', {'game_id': game_id})
                break
    else:
        socketio.emit('auth_response', {'success': False, 'error': 'Invalid credentials'})
        print(f"Socket authentication failed for {username}")


@app.route("/", methods=["GET", "POST"])
def home():
    players = load_data(PLAYER_DB)
    games = load_data(GAME_DB)
    username = session.get("user")

    # Check if user is already in a game
    if username:
        for game_id, game in games.items():
            if username in game["players"]:
                return redirect(url_for("playfield", game_id=game_id, player_id=username))

    # Handle login form submission
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in players and bcrypt.checkpw(password.encode(), players[username]["password"].encode()):
            # Set session with a unique identifier
            session["user"] = username
            session["login_time"] = uuid.uuid4().hex  # Add a unique token for this login session

            # Check if user is already in a game again (after login)
            for game_id, game in games.items():
                if username in game["players"]:
                    return redirect(url_for("playfield", game_id=game_id, player_id=username))

            return redirect(url_for("home"))
        return render_template("home.html", games=games, login_error="Invalid credentials")

    return render_template("home.html", games=games, username=username)


@app.route("/register", methods=["POST"])
def register():
    players = load_data(PLAYER_DB)
    username = request.form["username"]
    password = request.form["password"]

    if username in players:
        return render_template("home.html", register_error="Username already taken")

    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    players[username] = {"password": hashed_password, "wins": 0, "losses": 0}
    save_data(PLAYER_DB, players)

    # Set session variables
    session["user"] = username
    session["login_time"] = uuid.uuid4().hex

    return redirect(url_for("home"))


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


@app.route("/logout", methods=['POST'])
def logout(data=None):
    username = session.get("user")

    # Clear all socket connections for this user
    if username and username in user_socket_map:
        # Store sids in a list since we'll be modifying the set during iteration
        user_sids = list(user_socket_map[username])
        for sid in user_sids:
            if sid in socket_user_map:
                del socket_user_map[sid]
            # Disconnect the socket
            socketio.emit('force_disconnect', {'reason': 'User logged out'}, to=sid)
            try:
                socketio.server.disconnect(sid)
            except Exception as e:
                print(f"Error disconnecting socket {sid}: {str(e)}")

        # Remove user from mappings
        del user_socket_map[username]

    # Clear Flask session
    session.clear()

    return redirect(url_for("home"))


# Helper function to get active sockets for a username
def get_user_sockets(username):
    return user_socket_map.get(username, set())


# Helper to broadcast to all sockets belonging to a specific user
def emit_to_user(username, event, data):
    if username in user_socket_map:
        for sid in user_socket_map[username]:
            socketio.emit(event, data, to=sid)


@socketio.on('refresh_games')
def send_game_list(data=None):
    """Send game list to all clients"""
    socketio.emit('update_game_list', {"games": GamesKeeper().game_list()})


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
    for sid in user_socket_map[user_id]:
        join_room(game_id, sid)
    GamesKeeper().add_player(game_id, player)  # Should trigger a callback to EventBroadcaster
    print(f"User {user_id} joined game {game_id}")


@socketio.on('new_game')
def create_game(data=None):
    user = session.get('user')  # Retrieve user from session

    if not user:
        print("Unauthorized game creation attempt.")
        socketio.emit('error', {'message': 'You must be logged in to start a game'})
        return

    game_id = str(uuid.uuid4())
    GamesKeeper().add_game(game_id, GameWrapper(game_id, GameEventHandler(socketio, game_id)))
    # Notify the user and update the game list for all players
    socketio.emit('game_created', {'game_id': game_id})
    print(f"New game created by {user}: {game_id}")
    add_human_player(user, game_id)


@socketio.on('join_game')
def handle_join_game(data):
    user_id = session.get('user')

    if not user_id:
        socketio.emit('error', {'message': 'You must be logged in to join a game'})
        return

    game_id = data.get("game_id")
    add_human_player(user_id, game_id)


@socketio.on('add_ai_player')
def add_ai_player(data):
    user_id = session.get('user')
    game_id = GamesKeeper().find_owners_game(user_id)

    if not game_id:
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
    GamesKeeper().add_player(game_id, new_ai, opponent_index)
    send_game_state()


@socketio.on('view_game')
def view_game(data=None):
    # TODO: view game but can't interact with it (unless someone leaves)
    pass


# Add new socket event handler
@socketio.on('start_game')
def start_game(data=None):
    player_id = session.get('user')
    game_id = GamesKeeper().find_owners_game(player_id)

    if not game_id or game_id not in GamesKeeper().get_games():
        return {'error': 'Game not found'}

    GamesKeeper().get_game(game_id).start()


def find_valid_game(user_id, game_id=None):
    games = GamesKeeper().get_games()

    # Check if the provided game_id is valid
    if not game_id or game_id not in games:
        # Search for a valid game based on user_id
        game_id = next((gid for gid, gm in games.items() if user_id in GamesKeeper().get_player_names(gid)), None)

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
        return None  # Invalid game ID or player not in game

    gm = GamesKeeper().get_game(game_id)
    player_names = GamesKeeper().get_player_names(game_id)

    if user_id not in player_names:
        raise KeyError  # The check above should ensure this does not happen

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
                "status": gm.get_player_status(gm.players[opponent_index]),
            })
        else:
            opponent_details.append({"name": None, "card_count": 0, "status": "Absent"})

    # Use index because we need an exact match - suit is significant
    playable_indices = [c.cards[-1].get_index() for c in player.possible_plays(player.target_meld)[:-1]]

    playable_cards = []
    for card in player._hand:
        # Check each card to see if it's playable
        playable = card.get_index() in playable_indices
        playable_cards.append([card.get_value(), card.suit_str(), playable])

    return {
        "game_id": game_id,
        "player_id": user_id,
        "opponent_details": opponent_details,
        "is_owner": (gm.players[0].name == user_id),
        "player_hand": playable_cards
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
def send_game_state(data=None):
    """Send the full game state to the requesting player."""
    user_id = session.get('user')
    game_state = get_game_state(user_id)

    if not game_state:
        socketio.emit('error', {'message': 'Game not found or unauthorized access'})
        return

    # Emit the entire game state in one message
    emit_to_user(user_id, 'current_game_state', game_state)


@socketio.on('play_cards')
def handle_play_card(data):
    user_id = session.get('user')
    game_id = find_valid_game(user_id)

    if not game_id or game_id not in GamesKeeper().get_games():
        return {'error': 'Game not found'}

    game = GamesKeeper().get_game(game_id)
    if not game.episode:
        return {'error': 'Game not started'}
    if not game.episode.active_players:
        return {'error': 'Round not started'}
    if game.episode.active_players[0].name != user_id:
        return {'error': 'Not your turn'}

    # data['cards'] are a string like '5_0'
    meld = Meld()
    if data['cards'] != 'PASSED':
        for card in data['cards']:
            value, suit = card.split('_')
            meld = Meld(PlayingCard(int(value) * 4 + int(suit)), meld)

    # Process the play
    for p in game.players:
        if p.name == user_id:
            p.add_play(meld)
    game.episode.step()


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, use_reloader=False)
