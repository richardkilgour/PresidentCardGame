import bcrypt
from flask import request, session
from flask_socketio import join_room

from president.app.db import PLAYER_DB, load_data
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

        # If this user has a reserved slot, restore them and send them back to the game.
        reserved_game_id = GamesKeeper().find_reserved_game(username)
        if reserved_game_id:
            from president.app.game_persistence import save_game
            game = GamesKeeper().get_game(reserved_game_id)
            game.restore_human_player(username)
            game.clear_disconnect(username)
            join_room(reserved_game_id, sid)
            socketio.emit('rejoin_game', {'game_id': reserved_game_id}, to=sid)
            save_game(reserved_game_id)
        else:
            # Re-join game rooms, cancel any pending disconnect timeout, and
            # tell the client to navigate back (mirrors the reserved-slot path).
            for game_id in GamesKeeper().find_player(username):
                join_room(game_id, sid)
                game = GamesKeeper().get_game(game_id)
                game.clear_disconnect(username)
                socketio.emit('rejoin_game', {'game_id': game_id}, to=sid)
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
                _handle_player_fully_disconnected(username)

        del socket_user_map[sid]
        print(f"User {username} disconnected socket {sid}")
    else:
        print(f"Anonymous socket disconnected: {sid}")


def _handle_player_fully_disconnected(username: str) -> None:
    """Called when the last socket for a user closes."""
    from president.app.game_persistence import save_game

    for game_id in GamesKeeper().find_player(username):
        game = GamesKeeper().get_game(game_id)
        save_game(game_id)

        other_humans_connected = any(
            p.name != username and p.name in user_socket_map
            for _, p in game.human_players()
        )
        game.record_disconnect(username, other_humans_connected)
        socketio.emit('player_disconnected', {'username': username}, room=game_id)


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

        # Redirect to active game if one exists
        game_ids = GamesKeeper().find_player(username)
        reserved_game = GamesKeeper().find_reserved_game(username)
        active_game = (game_ids[0] if game_ids else None) or reserved_game
        if active_game:
            socketio.emit('active_game', {'game_id': active_game})
    else:
        socketio.emit('auth_response', {'success': False, 'error': 'Invalid credentials'})
        print(f"Socket authentication failed for {username}")
