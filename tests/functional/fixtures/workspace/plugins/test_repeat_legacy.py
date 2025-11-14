from typing import List, Optional


def _parse_options(option_string: Optional[str]) -> tuple[int, str]:
    default_suffix = "-repeat"
    count = 2
    if not option_string:
        return count, default_suffix

    options = {}
    for part in option_string.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            options[key.strip()] = value.strip()
        else:
            options[part] = ""

    if "n_variants" in options and options["n_variants"]:
        try:
            count = int(options["n_variants"])
        except ValueError as exc:
            raise ValueError(
                f"Invalid n_variants value for test_repeat: {options['n_variants']}"
            ) from exc
        if count < 1:
            raise ValueError("n_variants for test_repeat must be >= 1")

    suffix = options.get("suffix", default_suffix) or default_suffix
    return count, suffix


def transform(text, exclude_patterns=None, plugin_option=None):
    count, suffix = _parse_options(plugin_option)

    results = [text]
    for idx in range(1, count):
        if idx == 1:
            results.append(f"{text}{suffix}")
        else:
            results.append(f"{text}{suffix}-{idx}")
    return results


def get_available_option_values() -> Optional[List[str]]:
    """
    Return supported options; first option is the default string.
    """
    return [
        "n_variants=2",
        "n_variants=<int>,suffix=<suffix>",
    ]
