import uuid

from flask import session
from flask_socketio import join_room

from president.app.extensions import socketio
from president.app.game_event_handler import GameEventHandler
from president.app.game_helpers import add_human_player, find_valid_game, get_state_for_user
from president.app.game_keeper import GamesKeeper
from president.app.game_wrapper import GameWrapper
from president.app.session_manager import emit_to_user, user_socket_map
from president.players.AsyncPlayer import AsyncPlayer
from president.players.PlayerHolder import PlayerHolder
from president.players.PlayerSimple import PlayerSimple
from president.players.PlayerSplitter import PlayerSplitter


def _user_has_active_game(user_id: str) -> bool:
    return bool(GamesKeeper().find_player(user_id) or GamesKeeper().find_reserved_game(user_id))


@socketio.on('refresh_games')
def send_game_list(data=None):
    socketio.emit('update_game_list', {"games": GamesKeeper().game_list()})


@socketio.on('new_game')
def create_game(data=None):
    user = session.get('user')

    if not user:
        socketio.emit('error', {'message': 'You must be logged in to start a game'})
        return

    if _user_has_active_game(user):
        emit_to_user(user, 'error', {'message': 'You are already in a game. Quit first to create a new one.'})
        return

    game_id = str(uuid.uuid4())
    GamesKeeper().add_game(game_id, GameWrapper(game_id, GameEventHandler(socketio, game_id)))
    socketio.emit('game_created', {'game_id': game_id})
    print(f"New game created by {user}: {game_id}")
    add_human_player(user, game_id)


@socketio.on('join_game')
def handle_join_game(data):
    from president.app.game_persistence import save_game
    user_id = session.get('user')

    if not user_id:
        socketio.emit('error', {'message': 'You must be logged in to join a game'})
        return

    if _user_has_active_game(user_id):
        emit_to_user(user_id, 'error', {'message': 'You are already in a game. Quit first to join another.'})
        return

    game_id = data.get("game_id")
    if game_id not in GamesKeeper().get_games():
        emit_to_user(user_id, 'error', {'message': 'Game not found'})
        return

    game = GamesKeeper().get_game(game_id)

    # Pre-game join: game hasn't started yet
    if not game.episode:
        add_human_player(user_id, game_id)
        return

    # Mid-game join: take the first non-reserved AI seat
    ai_seat = next(
        (i for i, p in enumerate(game.player_manager.players)
         if p and not isinstance(p, AsyncPlayer) and i not in game.reserved_slots),
        None
    )
    if ai_seat is None:
        emit_to_user(user_id, 'error', {'message': 'No available seats in this game'})
        return

    old_player = game.player_manager.players[ai_seat]
    game.swap_player(old_player, AsyncPlayer(user_id))
    for sid in user_socket_map.get(user_id, set()):
        join_room(game_id, sid)
    save_game(game_id)


@socketio.on('quit_game')
def handle_quit_game(data=None):
    from president.app.game_persistence import delete_game, save_game
    user_id = session.get('user')
    if not user_id:
        return

    game_id = find_valid_game(user_id) or GamesKeeper().find_reserved_game(user_id)
    if not game_id:
        emit_to_user(user_id, 'quit_confirmed', {})
        return

    game = GamesKeeper().get_game(game_id)

    if GamesKeeper().find_reserved_game(user_id) == game_id:
        # User is quitting from a reserved slot — just clear the reservation
        for seat, name in list(game.reserved_slots.items()):
            if name == user_id:
                del game.reserved_slots[seat]
                break
    else:
        # Active human: replace with AI, not reserved (they quit permanently)
        game.replace_human_with_ai(user_id, reserved=False)

    game.clear_disconnect(user_id)
    socketio.emit('player_quit', {'username': user_id}, room=game_id)

    if not game.all_human_usernames():
        delete_game(game_id)
        GamesKeeper().remove_game(game_id)
    else:
        save_game(game_id)

    emit_to_user(user_id, 'quit_confirmed', {})


@socketio.on('replace_with_ai')
def handle_replace_with_ai(data):
    from president.app.game_persistence import save_game
    user_id = session.get('user')
    target = data.get('username')

    if not user_id or not target:
        return

    game_id = find_valid_game(user_id)
    if not game_id:
        return

    game = GamesKeeper().get_game(game_id)

    info = game.disconnect_info.get(target)
    if not info or not info.get('notified'):
        emit_to_user(user_id, 'error', {'message': 'Replacement not available yet'})
        return

    game.replace_human_with_ai(target, reserved=True)
    game.clear_disconnect(target)
    save_game(game_id)
    socketio.emit('player_replaced', {'username': target}, room=game_id)


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

    GamesKeeper().add_player(game_id, new_ai, opponent_index)
    send_game_state()


@socketio.on('view_game')
def view_game(data=None):
    # TODO: view game but can't interact with it (unless someone leaves)
    pass


# TODO: The timed start button hammers this.
@socketio.on('start_game')
def start_game(data=None):
    player_id = session.get('user')
    game_id = GamesKeeper().find_owners_game(player_id)

    if not game_id or game_id not in GamesKeeper().get_games():
        return {'error': 'Game not found'}

    game = GamesKeeper().get_game(game_id)
    if game.can_start():
        game.start()


@socketio.on('request_game_state')
def send_game_state(data=None):
    user_id = session.get('user')
    game_state = get_state_for_user(user_id)

    if not game_state:
        socketio.emit('error', {'message': 'Game not found or unauthorized access'})
        return

    emit_to_user(user_id, 'current_game_state', game_state)


@socketio.on('play_cards')
def handle_play_card(data):
    user_id = session.get('user')
    game_id = find_valid_game(user_id)

    if not game_id or game_id not in GamesKeeper().get_games():
        return {'error': 'Game not found'}

    game = GamesKeeper().get_game(game_id)
    error = game.play(user_id, data['cards'])
    if error:
        return {'error': error}
