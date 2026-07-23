"""Executable WSGI/Socket.IO entrypoint."""

from dancify import create_app
from dancify.extensions import socketio

app = create_app()

if __name__ == "__main__":
    socketio.run(  # pyright: ignore[reportUnknownMemberType]
        app, host="0.0.0.0", port=5000
    )
