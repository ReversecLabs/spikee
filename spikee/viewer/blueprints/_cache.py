# spikee/viewer/blueprints/_cache.py
"""Background module-tag cache.

Warmed in a daemon thread started from create_app() so that by the time the
user first navigates to /generate or /test the module tags are ready.

All mutable state (_store and _type_ready) is protected by a single lock
(_store_lock). _all_ready is a threading.Event and is inherently thread-safe.
"""

from __future__ import annotations

import sys
import threading

from spikee.utilities.modules import collect_modules

from spikee.viewer.blueprints._tags import _compute_tags

_MODULE_TYPES = ("plugins", "attacks", "targets", "judges")

# ── Internal state ────────────────────────────────────────────────────────────
# _store and _type_ready are both guarded by _store_lock.
# _all_ready is a threading.Event — its own internal lock makes it safe.

_store: dict[str, list[dict]] = {}  # "{module_type}/{name}" → [{label, colour}]
_store_lock = threading.Lock()
_type_ready: dict[str, bool] = {t: False for t in _MODULE_TYPES}
_all_ready = threading.Event()


# ── Internal helpers ──────────────────────────────────────────────────────────


def _evict_sys_modules(module_type: str) -> None:
    """Remove all cached entries for a module type from sys.modules."""
    prefix = f"spikee.{module_type}."
    to_delete = [
        k for k in sys.modules if k == f"spikee.{module_type}" or k.startswith(prefix)
    ]
    for key in to_delete:
        del sys.modules[key]


# ── Public API ────────────────────────────────────────────────────────────────


def warm_cache() -> None:
    """Load all module tags into _store. Meant to run in a daemon thread."""
    for module_type in _MODULE_TYPES:
        _evict_sys_modules(module_type)
        try:
            all_names, _, _ = collect_modules(module_type)
        except Exception:
            all_names = []
        for name in all_names:
            with _store_lock:
                _store[f"{module_type}/{name}"] = _compute_tags(name, module_type)
        with _store_lock:
            _type_ready[module_type] = True
    _all_ready.set()


def get_tags(name: str, module_type: str) -> list[dict] | None:
    """Return cached tags for *name*/*module_type*, or None if not yet warmed."""
    with _store_lock:
        return _store.get(f"{module_type}/{name}")


def is_type_ready(module_type: str) -> bool:
    """Return True if all tags for *module_type* have been loaded into cache."""
    with _store_lock:
        return _type_ready.get(module_type, False)


def is_ready() -> bool:
    """Return True if every module type has been fully cached."""
    return _all_ready.is_set()


def reset_cache() -> None:
    """Reset the cache entirely (used by the settings refresh endpoint)."""
    with _store_lock:
        _store.clear()
        for t in _type_ready:
            _type_ready[t] = False
    _all_ready.clear()


def status_dict() -> dict[str, bool]:
    """Return a snapshot of per-type readiness plus an 'all' key."""
    with _store_lock:
        d: dict[str, bool] = dict(_type_ready)
    d["all"] = _all_ready.is_set()
    return d
