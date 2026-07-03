"""Cached HTTP layer for the Sleeper tool.

Every outbound request goes through one ``requests_cache.CachedSession`` backed by SQLite, so we
(a) stay well under Sleeper's ~1000 req/min ceiling and (b) get deterministic, inspectable
responses during development. Per-URL TTLs encode the caching rules from CLAUDE.md:

- the ~5MB ``/players/nfl`` dump is refreshed once per day;
- weekly projections update through the week;
- league scoring/roster settings change rarely;
- live draft picks are **never** cached (we poll every ~3s during the draft).

The cache DB lives in ``data_cache/`` but is itself gitignored (it is volatile); the committed
parquet/SQLite data artifacts live alongside it.
"""

from __future__ import annotations

from pathlib import Path

from requests_cache import DO_NOT_CACHE, CachedSession

_HOUR = 3600
_DAY = 24 * _HOUR

# Repo root is two levels up from this file (src/sleeper/http.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DB = _REPO_ROOT / "data_cache" / "http_cache.sqlite"

DEFAULT_EXPIRE = _HOUR

# First matching pattern wins, so order most-specific first. Patterns match host + path.
URL_TTLS: dict[str, int] = {
    "api.sleeper.app/v1/draft/*": DO_NOT_CACHE,          # live draft polling
    "api.sleeper.app/v1/league/*/drafts": DO_NOT_CACHE,  # draft discovery (a draft can appear any minute on draft day)
    "api.sleeper.app/v1/players/nfl/trending/*": _HOUR,  # waiver trending signal, changes hourly
    "api.sleeper.app/v1/players/nfl*": _DAY,             # ~5MB dump, once/day
    "api.sleeper.com/projections/*": 6 * _HOUR,          # weekly projections
    "api.sleeper.app/v1/league/*": _HOUR,                # scoring/roster/rosters
}


def get_session(cache_db: Path | str = CACHE_DB) -> CachedSession:
    """Return the shared cached session used for every outbound Sleeper request.

    GET-only (the tool is strictly read-only). Stale responses are served if the network
    fails, so a flaky connection mid-week never blocks advice generation.
    """
    cache_db = Path(cache_db)
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    return CachedSession(
        # requests-cache appends ".sqlite"; strip it so we don't get http_cache.sqlite.sqlite.
        cache_name=str(cache_db.with_suffix("")),
        backend="sqlite",
        expire_after=DEFAULT_EXPIRE,
        urls_expire_after=URL_TTLS,
        stale_if_error=True,
        allowable_methods=("GET",),
    )
