# spikee/viewer/blueprints/_shared.py
"""Shared helpers for viewer blueprints — tag colours and module-tag loading."""

from __future__ import annotations

from spikee.utilities.enums import formatting_priority
from spikee.utilities.modules import get_description_from_module, load_module_from_path

# Bootstrap badge colour for each ModuleTag value.
# Single source of truth — imported by generate.py and test.py.
TAG_COLOURS: dict[str, str] = {
    "Encoding":           "secondary",
    "Formatting":         "secondary",
    "Obfuscation":        "secondary",
    "Social Engineering": "info",
    "Translation":        "info",
    "LLM":                "warning",
    "LLM-TTS":            "warning",
    "LLM-STT":            "warning",
    "LLM-STS":            "warning",
    "ML":                 "warning",
    "Image":              "primary",
    "Audio":              "primary",
    "Attack-Based":       "danger",
    "Multi-Turn":         "success",
    "Single-Turn":        "light",
}


def module_tags(name: str, module_type: str) -> list[dict]:
    """
    Load *name* as *module_type* and return its tags sorted by formatting_priority.

    Returns a list of ``{"label": str, "colour": str}`` dicts, or ``[]`` on any
    failure (missing dependency, legacy module without get_description, etc.).
    """
    try:
        mod = load_module_from_path(name, module_type)
        desc = get_description_from_module(mod, module_type)
        if desc and isinstance(desc, tuple) and desc[0]:
            sorted_tags = sorted(desc[0], key=formatting_priority)
            return [
                {
                    "label": tag.value if hasattr(tag, "value") else str(tag),
                    "colour": TAG_COLOURS.get(
                        tag.value if hasattr(tag, "value") else str(tag), "secondary"
                    ),
                }
                for tag in sorted_tags
            ]
    except Exception:
        pass
    return []
