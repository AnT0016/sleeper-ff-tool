"""Sleeper API client and the single cached HTTP layer all outbound calls go through.

Sleeper's endpoints are unofficial, undocumented, and may change — keep every call
isolated in this package (see CLAUDE.md > "Data sources & hard rules").
"""

from .http import get_session

__all__ = ["get_session"]
