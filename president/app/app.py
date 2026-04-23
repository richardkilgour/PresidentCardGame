from president.app.extensions import app, socketio
from president.app.scheduler import start_scheduler

start_scheduler()

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, use_reloader=False, host="0.0.0.0")
