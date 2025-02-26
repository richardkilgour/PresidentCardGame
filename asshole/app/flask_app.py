import importlib
import os
from uuid import uuid4

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import bcrypt

from asshole.core.GameMaster import GameMaster

app = Flask(__name__)
app.secret_key = "your_secret_key"
socketio = SocketIO(app, cors_allowed_origins="*")

PLAYER_DB = "db/players.json"
GAME_DB = "db/games.json"

games = {}
# Store player sessions
player_sessions = {}



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


@app.route("/", methods=["GET", "POST"])
def home():
    players = load_data(PLAYER_DB)
    games = load_data(GAME_DB)
    username = session.get("user")

    for game_id, game in games.items():
        if username in game["players"]:
            return redirect(url_for("playfield", game_id=game_id, player_id=username))

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in players and bcrypt.checkpw(password.encode(), players[username]["password"].encode()):
            session["user"] = username
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
    session["user"] = username
    return redirect(url_for("home"))


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))


@socketio.on('join_game')
def handle_join_game(data):
    player_id = request.args.get('player_id')

    # Try to find an available game
    available_game = None
    for game in games.values():
        if len(games["players"]) < 4:
            available_game = game
            break

    # Create new game if none available
    if not available_game:
        game_id = str(uuid4())
        games[game_id] = GameMaster()
        available_game = games[game_id]

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


# Add new socket event handler
@socketio.on('ready_to_start')
def handle_ready_to_start():
    player_id = request.args.get('player_id')
    game_id = player_sessions.get(player_id)

    if not game_id or game_id not in games:
        return {'error': 'Game not found'}

    game = games[game_id]
    # Make some computer players to round things out
    while len(game.players) < 4:
        module_name = 'players.PlayerSplitter'
        # Dynamically import the module
        module = importlib.import_module(module_name)
        player_class = getattr(module, 'PlayerSplitter')
        game.make_player(player_class, 'PlayerSplitter')
    # This is now blocking!
    game.start(number_of_rounds=100)


    socketio.emit('game_start', {
        'game_id': game.game_id,
        'first_player': game.current_turn
    }, room=game_id)


@socketio.on('play_cards')
def handle_play_cards(data):
    player_id = request.args.get('player_id')
    game_id = player_sessions.get(player_id)

    if not game_id or game_id not in games:
        return {'error': 'Game not found'}

    game = games[game_id]
    # TODO: Fix this
    if not game.is_player_turn(player_id):
        return {'error': 'Not your turn'}

    # Process the play
    card_id = data.get('cards')
    # Your card playing logic here

    # Update game state
    game.current_turn = (game.current_turn + 1) % len(game.players)

    # Broadcast the play to all players in the game
    socketio.emit('card_played', {
        'player_id': player_id,
        'card_id': card_id,
        'next_player': game.players[game.current_turn]
    }, room=game_id)


@socketio.on('disconnect')
def handle_disconnect():
    player_id = request.args.get('player_id')
    if player_id in player_sessions:
        game_id = player_sessions[player_id]
        if game_id:
            leave_room(game_id)
            # Notify other players but keep the game state
            socketio.emit('player_disconnected', {
                'player_id': player_id
            }, room=game_id)



@app.route("/join_game/<game_id>")
def join_game(game_id):
    games = load_data(GAME_DB)
    username = session.get("user", "Guest")

    if game_id in games and len(games[game_id]["players"]) < 4:
        games[game_id]["players"].append(username)
        save_data(GAME_DB, games)
        return redirect(url_for("playfield", game_id=game_id, player_id=username))

    # Send initial cards to the player - suit, index, playable
    socketio.emit('update_cards', {
        'cards': [[[0, 1, 1],[0, 1, 1],[0, 1, 1]],[[0, 1, 1],[0, 1, 1],[0, 1, 1]],[[0, 1, 1],[0, 1, 1],[0, 1, 1]],[[0, 1, 1],[0, 1, 1],[0, 1, 1]]]
    }, room=request.sid)


    return redirect(url_for("home"))


@app.route("/start_game")
def start_game():
    username = session.get("user")
    if not username:
        return redirect(url_for("home"))

    games = load_data(GAME_DB)
    game_id = uuid4()
    games[str(game_id)] = {"players": [username], "state": {}}
    save_data(GAME_DB, games)

    return redirect(url_for("playfield", game_id=game_id))


@app.route("/playfield/<game_id>")
def playfield(game_id):
    username = session.get("user")
    if not username:
        return redirect(url_for("home"))

    games = load_data(GAME_DB)
    if game_id not in games:
        return redirect(url_for("home"))

    players = games[game_id]["players"]

    # Determine if user is the host BEFORE reordering
    is_host = username == players[0]

    # Reorder players list to put the current user at index [0]
    if username in players:
        user_index = players.index(username)
        players = players[user_index:] + players[:user_index]

    # Ensure a GameMaster instance exists for this game
    if game_id not in games:
        games[game_id] = GameMaster()

    game_master = games[game_id]

    allCards = [[]] * 4

    return render_template(
        "game.html",
        game_id=game_id,
        current_player=username,
        # TODO: Make this persistent
        player_id=str(uuid4()),
        players=players,
        allCards=allCards,
        is_host=is_host
    )

# WebSocket: Handle player moves
@socketio.on("make_move")
def handle_move(data):
    game_id = data["game_id"]
    player_id = data["player_id"]
    move = data["move"]

    gm = games.get(game_id)
    if gm:
        next_player = gm.make_move(player_id, move)
        emit("move_made", {"player_id": player_id, "move": move}, room=game_id)
        emit("your_turn", {"player_id": next_player}, room=game_id)

# WebSocket: Handle player joining a game
@socketio.on("join_game")
def join_game(data):
    username = session.get("user")
    game_id = data["game_id"]

    join_room(game_id)
    emit("player_joined", {"player_id": username}, room=game_id)


@socketio.on("leave_game")
def leave_game(game_id):
    username = session.get("user")
    if not username:
        return redirect(url_for("home"))

    games = load_data(GAME_DB)
    if game_id not in games:
        return redirect(url_for("home"))

    players = games[game_id]["players"]

    if username in players:
        players.remove(username)
        players.append("AI_Placeholder")

        game_master = games[game_id]
        game_master.make_player("AI", "AI_Placeholder")

        save_data(GAME_DB, games)

    return redirect(url_for("playfield", game_id=game_id, player_id=username))


@app.route("/add_ai/<game_id>", methods=["POST"])
def add_ai(game_id):
    username = session.get("user")
    if not username:
        return redirect(url_for("home"))

    games = load_data(GAME_DB)
    if game_id not in games or username != games[game_id]["players"][0]:  # Ensure only the host can add AI
        return redirect(url_for("playfield", game_id=game_id, player_id=username))

    ai_type = request.form["ai_type"]
    ai_name = f"{ai_type.capitalize()} AI {len(games[game_id]['players'])}"

    if len(games[game_id]["players"]) < 4:
        games[game_id]["players"].append(ai_name)
        save_data(GAME_DB, games)

    return redirect(url_for("playfield", game_id=game_id, player_id=username))


if __name__ == "__main__":
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)

