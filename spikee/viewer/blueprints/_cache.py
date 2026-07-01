# spikee/viewer/blueprints/_cache.py
"""Background module-tag cache.

Warmed in a daemon thread started from create_app() so that by the time the
user first navigates to /generate or /test the module tags are ready.
"""
from __future__ import annotations

import threading

from spikee.utilities.modules import collect_modules

from spikee.viewer.blueprints._tags import _compute_tags

_MODULE_TYPES = ("plugins", "attacks", "targets", "judges")

# ── Internal state ────────────────────────────────────────────────────────────

_store: dict[str, list[dict]] = {}          # "{module_type}/{name}" → [{label, colour}]
_store_lock = threading.Lock()
_type_ready: dict[str, bool] = {t: False for t in _MODULE_TYPES}
_all_ready = threading.Event()


# ── Public API ────────────────────────────────────────────────────────────────

def _evict_sys_modules(module_type: str) -> None:
    """Remove all cached entries for a module type from sys.modules."""
    import sys
    prefix = f"spikee.{module_type}."
    to_delete = [k for k in sys.modules if k == f"spikee.{module_type}" or k.startswith(prefix)]
    for key in to_delete:
        del sys.modules[key]


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
    """Return cached tags, or None if this entry has not been warmed yet."""
    with _store_lock:
        return _store.get(f"{module_type}/{name}")


def is_type_ready(module_type: str) -> bool:
    with _store_lock:
        return _type_ready.get(module_type, False)


def is_ready() -> bool:
    return _all_ready.is_set()


def status_dict() -> dict:
    with _store_lock:
        d = dict(_type_ready)
    d["all"] = _all_ready.is_set()
    return d
