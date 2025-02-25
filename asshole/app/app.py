import json
import os
import uuid
from datetime import timedelta

from flask import Flask, request, redirect, session, render_template, url_for, flash, jsonify
from flask_socketio import SocketIO, join_room, leave_room

from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.permanent_session_lifetime = timedelta(days=7)
socketio = SocketIO(app, cors_allowed_origins="*", manage_session=True)

PLAYER_DB = "db/players.json"
GAME_DB = "db/games.json"

# Store player sessions
connected_users = set()


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
# Dictionary of games and the player IDs
games = load_data(GAME_DB)


@socketio.on('connect')
def handle_connect():
    # Get player_id from session
    player_id = session.get('user')
    connected_users.add(player_id)
    # TODO: If this user is currently in a game, send them there
    return True

@socketio.on('disconnect')
def handle_connect():
    # Get player_id from session
    player_id = session.get('user')
    if player_id in connected_users:
        connected_users.remove(player_id)
    # TODO: If this was in a game, send notify other players
    return True

@app.route('/')
def index():
    # TODO: If this user is currently in a game, send them there
    return render_template(
        'home.html',
        games=games,
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
    session['logged_in'] = False
    if user:
        session.pop('user')
    # Save the session
    session.modified = True

# Also allow log out via HTTP
@app.route('/logout', methods=['POST'])
def http_logout():
    session['logged_in'] = False
    if 'user' in session:
        session.pop('user')
    return redirect(url_for('index'))


@socketio.on('refresh_games')
def send_game_list():
    """Send game list to all clients"""
    game_list = [{"id": game_id, "players": players} for game_id, players in games.items()]
    socketio.emit('update_game_list', {"games": game_list})


@app.route('/get_user_info')
def get_user_info():
    if 'logged_in' in session and session['logged_in']:
        info = {"logged_in": True, "username": session.get('user')}
    else:
        info = {"logged_in": False}
    return jsonify(info)


@socketio.on('new_game')
def create_new_game():
    user = session.get('user')  # Retrieve user from session

    if not user:
        print("Unauthorized game creation attempt.")
        socketio.emit('error', {'message': 'You must be logged in to start a game'})
        return

    new_game_id = str(uuid.uuid4())
    games[new_game_id] = [user]

    print(f"New game created by {user}: {new_game_id}")
    # TODO Save games
    save_data(GAME_DB, games)

    # Notify the user and update the game list for all players
    socketio.emit('game_created', {'game_id': new_game_id})
    send_game_list()

@socketio.on('join_game')
def handle_join_game(data):
    player_id = request.args.get('player_id')

    # Try to find an available game
    available_game = None
    for game in games.values():
        if len(games["players"]) < 4:
            available_game = game
            break

    # Add player to game
    # TODO: Should be an AbstractPlayer
    if available_game.add_player(player_id):
        join_room(available_game.id)
        player_sessions[player_id] = available_game.id

        # Notify all players in the game
        socketio.emit('player_joined', {
            'game_id': available_game.id,
            'player_count': len(available_game.players)
        }, room=available_game.id)

        # Start game if ready
        if len(games["players"]) == 4:
            socketio.emit('game_start', {
                'game_id': available_game.id,
                'first_player': available_game.players[0]
            }, room=available_game.id)


@socketio.on('view_game')
def view_game():
    # TODO: view game but can't interact with it (unless someone leaves)
    pass

# Add new socket event handler
@socketio.on('ready_to_start')
def handle_ready_to_start():
    player_id = request.args.get('player_id')
    game_id = player_sessions.get(player_id)

    if not game_id or game_id not in games:
        return {'error': 'Game not found'}

    game = games[game_id]
    all_ready = game.player_ready(player_id)

    if all_ready:
        game.state = "active"
        game.current_turn = list(game.players.keys())[0]
        socketio.emit('game_start', {
            'game_id': game.id,
            'first_player': game.current_turn
        }, room=game_id)
    else:
        socketio.emit('player_ready', {
            'player_id': player_id,
            'ready_count': len(game.ready_players),
            'total_players': len(game.players)
        }, room=game_id)


@app.route('/game/<game_id>')
def game(game_id):
    if not session.get('logged_in', False):
        # TODO: allow anonymous players
        return redirect(url_for('index'))

    user = session.get('user')
    players = games[game_id]
    if user in players:
        player_index = players.index(user)
        # TODO: Check this logic
        players = players[player_index:] + players[0:player_index]

    # Dummy Game
    # Send the player's hand
    socketio.emit('notify_current_hand', {
        # TODO: Get the player's cards
        'playerCards': [[1,1,1], [1,1,1], [1,1,1], [1,1,1],[1,1,1],[1,1,1]]
    })

    # Send the opponent's card count
    socketio.emit('notify_opponent_hands', {
        # TODO: Get the other players' card counts
        'opponent_cards': [1, 2, 3] # opponent_card_counts
    })


    return render_template('game.html')


@socketio.on('play_card')
def handle_play_card(data):
    player_id = request.args.get('player_id')
    game_id = player_sessions.get(player_id)

    if not game_id or game_id not in games:
        return {'error': 'Game not found'}

    game = games[game_id]
    if not game.is_player_turn(player_id):
        return {'error': 'Not your turn'}

    # Process the play
    card_id = data.get('card_id')
    # Your card playing logic here

    # Update game state
    game.current_turn = (game.current_turn + 1) % len(game.players)

    # Broadcast the play to all players in the game
    socketio.emit('card_played', {
        'player_id': player_id,
        'card_id': card_id,
        'next_player': game.players[game.current_turn]
    }, room=game_id)


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)