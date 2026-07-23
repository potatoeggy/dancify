"""Application extension instances."""

from flask_socketio import SocketIO  # type: ignore[import-untyped]
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)
socketio = SocketIO(cors_allowed_origins=[], async_mode="threading")
