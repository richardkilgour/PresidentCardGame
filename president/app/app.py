from president.app.extensions import app, socketio
from president.app.scheduler import start_scheduler

# Blueprints group routes defined in other modules. Importing the module creates the
# Blueprint object (the @blueprint.route decorators run at import time), but the routes
# don't exist on the app yet. register_blueprint() is the step that actually wires them
# in — Flask copies each blueprint's URL rules into the app's routing table here.
from president.app.auth import auth_bp
from president.app.game_routes import game_routes_bp

app.register_blueprint(auth_bp)
app.register_blueprint(game_routes_bp)

# SocketIO event handlers have no blueprint equivalent — they register directly on the
# socketio singleton via @socketio.on(). We just need to import these modules so their
# decorators run before the server starts. Without these imports the handlers are never
# registered and the server silently ignores all incoming socket events.
import president.app.game_events  # noqa: F401
import president.app.socket_auth  # noqa: F401

start_scheduler()

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, use_reloader=False, host="0.0.0.0")
