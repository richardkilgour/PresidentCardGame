from president.app.extensions import socketio

user_socket_map = {}  # {username: {sid1, sid2, ...}}
socket_user_map = {}  # {sid: username}


def get_user_sockets(username):
    return user_socket_map.get(username, set())


def emit_to_user(username, event, data):
    if username in user_socket_map:
        for sid in user_socket_map[username]:
            socketio.emit(event, data, to=sid)
