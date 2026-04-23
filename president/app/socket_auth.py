import bcrypt
from flask import request, session
from flask_socketio import join_room

from president.app.db import GAME_DB, PLAYER_DB, load_data
from president.app.extensions import socketio
from president.app.game_keeper import GamesKeeper
from president.app.session_manager import socket_user_map, user_socket_map


@socketio.on('connect')
def handle_connect(data=None):
    username = session.get("user")
    sid = request.sid

    if username:
        if username not in user_socket_map:
            user_socket_map[username] = set()
        user_socket_map[username].add(sid)
        socket_user_map[sid] = username

        socketio.emit('connection_status', {'status': 'connected', 'username': username})
        print(f"User {username} connected with socket ID: {sid}")

        for game_id in GamesKeeper().find_player(username):
            for s in user_socket_map[username]:
                join_room(game_id, s)
    else:
        socketio.emit('connection_status', {'status': 'anonymous'})
        print(f"Anonymous socket connection: {sid}")


@socketio.on('disconnect')
def handle_disconnect(data=None):
    sid = request.sid
    username = socket_user_map.get(sid)

    if username:
        if username in user_socket_map:
            user_socket_map[username].discard(sid)
            if not user_socket_map[username]:
                del user_socket_map[username]

        del socket_user_map[sid]
        print(f"User {username} disconnected socket {sid}")
    else:
        print(f"Anonymous socket disconnected: {sid}")


@socketio.on('authenticate')
def handle_socket_authentication(data):
    sid = request.sid
    username = data.get('username')
    password = data.get('password')

    players = load_data(PLAYER_DB)

    if username in players and bcrypt.checkpw(password.encode(), players[username]["password"].encode()):
        session["user"] = username
        session.modified = True

        if username not in user_socket_map:
            user_socket_map[username] = set()
        user_socket_map[username].add(sid)
        socket_user_map[sid] = username

        socketio.emit('auth_response', {'success': True, 'username': username})
        print(f"Socket authentication successful for {username}")

        games = load_data(GAME_DB)
        for game_id, game in games.items():
            if username in game["players"]:
                socketio.emit('active_game', {'game_id': game_id})
                break
    else:
        socketio.emit('auth_response', {'success': False, 'error': 'Invalid credentials'})
        print(f"Socket authentication failed for {username}")
