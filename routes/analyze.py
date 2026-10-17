"""Compatibility import. All routes are registered once by the application factory."""
from app import app, create_app

__all__ = ["app", "create_app"]
