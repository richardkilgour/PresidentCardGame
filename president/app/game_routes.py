from flask import redirect, render_template, session, url_for

from president.app.extensions import app
from president.app.game_helpers import get_state_for_user


@app.route('/game/<game_id>')
def show_game(game_id):
    if not session.get('logged_in', False):
        return redirect(url_for('index'))

    user = session.get('user')
    game_state = get_state_for_user(user, game_id)

    if not game_state:
        return "Game not found or unauthorized access", 404

    return render_template('game.html',
                           player_names=game_state["player_names"],
                           game_id=game_state["game_id"],
                           opponent_cards=game_state["opponent_cards"],
                           is_owner=game_state["is_owner"])
