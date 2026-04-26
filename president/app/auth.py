from __future__ import annotations

import uuid

import bcrypt
from flask import Blueprint, jsonify, redirect, render_template, request, session, url_for
from president.app.db import PLAYER_DB, load_data, save_data
from president.app.extensions import socketio
from president.app.game_keeper import GamesKeeper
from president.app.session_manager import socket_user_map, user_socket_map

auth_bp = Blueprint('auth', __name__)


def _active_game_for(username: str) -> str | None:
    """Return the game_id the user should be redirected to, or None."""
    game_ids = GamesKeeper().find_player(username)
    if game_ids:
        return game_ids[0]
    return GamesKeeper().find_reserved_game(username)


@auth_bp.route("/", methods=["GET", "POST"])
def home():
    players = load_data(PLAYER_DB)
    username = session.get("user")

    if username:
        active_game = _active_game_for(username)
        if active_game:
            return redirect(url_for("game_routes.show_game", game_id=active_game))

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in players and bcrypt.checkpw(password.encode(), players[username]["password"].encode()):
            session["user"] = username
            session["logged_in"] = True
            session["login_time"] = uuid.uuid4().hex

            active_game = _active_game_for(username)
            if active_game:
                return redirect(url_for("game_routes.show_game", game_id=active_game))

            return redirect(url_for("auth.home"))
        return render_template("home.html", login_error="Invalid credentials")

    return render_template("home.html", username=username)


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


@auth_bp.route('/api/anonymous_login', methods=['POST'])
def anonymous_login():
    """Create a temporary session for clients that don't have registered credentials."""
    data = request.get_json() or {}
    name = (data.get('name') or '').strip()
    if not name:
        name = f"Guest_{uuid.uuid4().hex[:6]}"
    display_name = f"{name} (unregistered)"
    session['user'] = display_name
    session['logged_in'] = True
    return jsonify({'success': True, 'username': display_name})


@auth_bp.route('/api/games', methods=['GET'])
def api_games():
    """HTTP endpoint returning the current game list (mirrors the SocketIO refresh_games event)."""
    from president.app.game_keeper import GamesKeeper
    return jsonify({'games': GamesKeeper().game_list()})
