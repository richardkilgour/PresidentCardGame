import uuid

import bcrypt
from flask import Blueprint, flash, jsonify, redirect, render_template, request, session, url_for
from president.app.db import GAME_DB, PLAYER_DB, load_data, save_data
from president.app.extensions import socketio
from president.app.session_manager import socket_user_map, user_socket_map

# Blueprint objects are created here, in the module that owns the routes. The 'auth'
# string is the blueprint's internal name — Flask uses it as a prefix for url_for()
# lookups, so url_for('auth.home') resolves to this module's home() view. __name__
# tells Flask where to find templates and static files relative to this module.
auth_bp = Blueprint('auth', __name__)


@auth_bp.route("/", methods=["GET", "POST"])
def home():
    players = load_data(PLAYER_DB)
    games = load_data(GAME_DB)
    username = session.get("user")

    if username:
        for game_id, game in games.items():
            if username in game["players"]:
                return redirect(url_for("game_routes.show_game", game_id=game_id))

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in players and bcrypt.checkpw(password.encode(), players[username]["password"].encode()):
            session["user"] = username
            session["logged_in"] = True
            session["login_time"] = uuid.uuid4().hex

            for game_id, game in games.items():
                if username in game["players"]:
                    return redirect(url_for("game_routes.show_game", game_id=game_id))

            return redirect(url_for("auth.home"))
        return render_template("home.html", games=games, login_error="Invalid credentials")

    return render_template("home.html", games=games, username=username)


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    players = load_data(PLAYER_DB)
    if username in players:
        return jsonify({"success": False, "error": "Username already taken"})

    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    players[username] = {"password": hashed_password, "wins": 0, "losses": 0}
    save_data(PLAYER_DB, players)

    session["user"] = username
    session["logged_in"] = True
    session["login_time"] = uuid.uuid4().hex

    return jsonify({"success": True})


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username, password = data.get('username'), data.get('password')

    players = load_data(PLAYER_DB)
    if username in players and bcrypt.checkpw(password.encode(), players[username]["password"].encode()):
        session['user'] = username
        session['logged_in'] = True
        return jsonify({"success": True})

    return jsonify({"success": False, "error": "Invalid credentials"})


@auth_bp.route("/logout", methods=['POST'])
def logout(data=None):
    username = session.get("user")

    if username and username in user_socket_map:
        user_sids = list(user_socket_map[username])
        for sid in user_sids:
            if sid in socket_user_map:
                del socket_user_map[sid]
            socketio.emit('force_disconnect', {'reason': 'User logged out'}, to=sid)
            try:
                socketio.server.disconnect(sid)
            except Exception as e:
                print(f"Error disconnecting socket {sid}: {str(e)}")

        del user_socket_map[username]

    session.clear()
    return redirect(url_for("auth.home"))


@auth_bp.route('/get_user_info')
def get_user_info():
    if 'logged_in' in session and session['logged_in']:
        info = {"logged_in": True, "username": session.get('user')}
    else:
        info = {"logged_in": False}
    return jsonify(info)
