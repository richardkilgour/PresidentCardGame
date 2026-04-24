import uuid

from flask import session

from president.app.extensions import socketio
from president.app.game_event_handler import GameEventHandler
from president.app.game_helpers import add_human_player, find_valid_game, get_state_for_user
from president.app.game_keeper import GamesKeeper
from president.app.game_wrapper import GameWrapper
from president.app.session_manager import emit_to_user
from president.players.PlayerHolder import PlayerHolder
from president.players.PlayerSimple import PlayerSimple
from president.players.PlayerSplitter import PlayerSplitter


@socketio.on('refresh_games')
def send_game_list(data=None):
    socketio.emit('update_game_list', {"games": GamesKeeper().game_list()})


@socketio.on('new_game')
def create_game(data=None):
    user = session.get('user')

    if not user:
        print("Unauthorized game creation attempt.")
        socketio.emit('error', {'message': 'You must be logged in to start a game'})
        return

    game_id = str(uuid.uuid4())
    GamesKeeper().add_game(game_id, GameWrapper(game_id, GameEventHandler(socketio, game_id)))
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
