from typing import List, Optional

from spikee.utilities.modules import parse_options


def transform(text, exclude_patterns=None, plugin_option=None):
    options = parse_options(plugin_option)
    count = int(options.get("n_variants", 2))
    suffix = options.get("suffix", "-repeat")

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
