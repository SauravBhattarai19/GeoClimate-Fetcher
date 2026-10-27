"""
Session-cached Earth Engine tile URLs for folium maps.

getMapId() URLs are not permanent — GEE expires the map ID after a while, after
which every tile 404s and the layer silently goes blank. Entries are therefore
re-registered once they are older than TILE_URL_TTL_S.

Only memory-limit failures are cached (re-running the same expression will fail
the same way). Transient errors — network blips, 429 rate limits, quota spikes —
are not cached, so the next re-render retries instead of locking the layer out
for the rest of the session.
"""

import time

import streamlit as st

TILE_URL_TTL_S = 60 * 60  # re-register map IDs hourly, well inside GEE's expiry
TILE_FAILED = "__FAILED__"  # sentinel cached for permanent (memory-limit) failures

_PERMANENT_ERROR_MARKERS = ("memory limit", "memory capacity", "out of memory")


def _is_permanent_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _PERMANENT_ERROR_MARKERS)


def has_cached_tile_url(cache_name: str, cache_key: str) -> bool:
    """True when a fresh, successful tile URL is cached for cache_key."""
    entry = st.session_state.get(cache_name, {}).get(cache_key)
    return (
        isinstance(entry, dict)
        and time.time() - entry["ts"] < TILE_URL_TTL_S
    )


def get_ee_tile_url(image, vis_params: dict, cache_name: str, cache_key: str,
                    failed_msg: str) -> str:
    """
    Register an ee.Image with GEE and cache its tile URL in
    st.session_state[cache_name][cache_key].

    Returns the cached URL on re-renders until it is TILE_URL_TTL_S old.
    Raises RuntimeError(failed_msg) for a cached memory-limit failure, and
    re-raises (uncached) for any other error.
    """
    cache = st.session_state.setdefault(cache_name, {})
    entry = cache.get(cache_key)

    if entry == TILE_FAILED:
        raise RuntimeError(failed_msg)
    if has_cached_tile_url(cache_name, cache_key):
        return entry["url"]

    try:
        map_id = image.getMapId(vis_params)
    except Exception as exc:
        if _is_permanent_error(exc):
            cache[cache_key] = TILE_FAILED
        else:
            cache.pop(cache_key, None)
        raise RuntimeError(str(exc)) from exc

    url = map_id["tile_fetcher"].url_format
    cache[cache_key] = {"url": url, "ts": time.time()}
    return url
