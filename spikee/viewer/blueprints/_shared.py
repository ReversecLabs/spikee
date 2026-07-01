# spikee/viewer/blueprints/_shared.py
"""Shared helpers for viewer blueprints — module-tag loading."""

from __future__ import annotations

from spikee.utilities.modules import load_module_from_path

from spikee.viewer.blueprints._tags import _get_tag_label, _get_tag_colour


def module_tags(name: str, module_type: str) -> list[dict]:
    """Return sorted [{label, colour}] tag list for *name* / *module_type*.
    
    Checks the background cache first; falls back to a live load on cache miss
    (covers the startup race window before warm_cache() has reached this entry).
    """
    from spikee.viewer.blueprints import _cache  # local import avoids circular

    cached = _cache.get_tags(name, module_type)
    if cached is not None:
        return cached

    # Cache miss — live load via _compute_tags from _tags module
    return _compute_tags_live(name, module_type)


def _compute_tags_live(name: str, module_type: str) -> list[dict]:
    """Live tag computation for cache-miss fallback.
    
    Loads the module directly and extracts its DESCRIPTION tuple.
    Returns empty list on any error.
    """
    try:
        mod = load_module_from_path(name, module_type)
        desc = mod.DESCRIPTION if hasattr(mod, "DESCRIPTION") else None
        if desc and isinstance(desc, tuple) and desc[0]:
            from spikee.utilities.enums import formatting_priority

            sorted_tags = sorted(desc[0], key=formatting_priority)
            return [
                {"label": _get_tag_label(tag), "colour": _get_tag_colour(tag)}
                for tag in sorted_tags
            ]
    except Exception:
        pass
    return []
