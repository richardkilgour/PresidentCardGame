import os

os.environ['PRESIDENT_SCHEDULER_INTERVAL'] = '0.05'

import threading
import time

import pytest

from president.app.extensions import app, socketio

TEST_PORT = 5099


@pytest.fixture(scope="session", autouse=True)
def live_server():
    import president.app.app  # noqa — registers blueprints, starts scheduler

    t = threading.Thread(
        target=lambda: socketio.run(
            app,
            host="localhost",
            port=TEST_PORT,
            use_reloader=False,
            allow_unsafe_werkzeug=True,
        ),
        daemon=True,
    )
    t.start()
    time.sleep(0.5)
    yield f"http://localhost:{TEST_PORT}"
