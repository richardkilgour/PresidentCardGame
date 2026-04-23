import os
import warnings
from datetime import timedelta

from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)

secret_key = os.environ.get("SECRET_KEY")
if not secret_key:
    warnings.warn("SECRET_KEY env var not set — using insecure dev default", stacklevel=2)
    secret_key = "dev-only-do-not-use-in-production"
app.secret_key = secret_key

app.permanent_session_lifetime = timedelta(days=7)
socketio = SocketIO(app, cors_allowed_origins="*", manage_session=True)
