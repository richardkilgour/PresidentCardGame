from flask import Blueprint, redirect, render_template, session, url_for

from president.app.game_helpers import get_state_for_user

# Routes for serving game pages live here. The blueprint is registered in app.py,
# which is when these URL rules are added to Flask's routing table.
game_routes_bp = Blueprint('game_routes', __name__)


@game_routes_bp.route('/game/<game_id>')
def show_game(game_id):
    if not session.get('logged_in', False):
        return redirect(url_for('auth.home'))

    user = session.get('user')
    game_state = get_state_for_user(user, game_id)

    if not game_state:
        return "Game not found or unauthorized access", 404

    return render_template('game.html',
                           player_names=game_state["player_names"],
                           game_id=game_state["game_id"],
                           opponent_cards=game_state["opponent_cards"],
                           is_owner=game_state["is_owner"])
