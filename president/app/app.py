import argparse

from president.app.extensions import app, socketio
from president.app.scheduler import start_scheduler

from president.app.auth import auth_bp
from president.app.game_routes import game_routes_bp

app.register_blueprint(auth_bp)
app.register_blueprint(game_routes_bp)

import president.app.game_events  # noqa: F401
import president.app.socket_auth  # noqa: F401

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', action='store_true', help='Start fresh, ignoring any saved games')
    args = parser.parse_args()

    if not args.clean:
        from president.app.game_persistence import load_all_games
        load_all_games()

    from president.app.game_defaults import seed_default_games
    seed_default_games()

    start_scheduler()
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, use_reloader=False, host="0.0.0.0")
