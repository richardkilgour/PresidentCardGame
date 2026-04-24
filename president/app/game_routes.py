from flask import Blueprint, redirect, render_template, session, url_for

from president.app.game_helpers import get_state_for_user
from president.app.game_keeper import GamesKeeper

game_routes_bp = Blueprint('game_routes', __name__)


@game_routes_bp.route('/game/<game_id>')
def show_game(game_id):
    if not session.get('logged_in', False):
        return redirect(url_for('auth.home'))

    user = session.get('user')

    # If the user has a reserved slot in this game, restore them before building state.
    # (The socket connect handler does the same thing but fires after the HTTP response.)
    games = GamesKeeper().get_games()
    if game_id in games:
        game = games[game_id]
        if user in game.reserved_slots.values():
            from president.app.game_persistence import save_game
            game.restore_human_player(user)
            game.clear_disconnect(user)
            save_game(game_id)

    game_state = get_state_for_user(user, game_id)

    if not game_state:
        return "Game not found or unauthorized access", 404

    return render_template('game.html',
                           player_names=game_state["player_names"],
                           game_id=game_state["game_id"],
                           opponent_cards=game_state["opponent_cards"],
                           is_owner=game_state["is_owner"])
