from typing import List, Optional, Union

from spikee.templates.plugin import Plugin
from spikee.utilities.modules import parse_options


class TestRepeatPlugin(Plugin):
    DEFAULT_SUFFIX = "-repeat"
    DEFAULT_COUNT = 2

    def get_available_option_values(self) -> List[str]:
        return [
            "n_variants=2",
            "n_variants=<int>,suffix=<suffix>",
        ]

    def transform(
        self,
        text: str,
        exclude_patterns: Optional[List[str]] = None,
        plugin_option: Optional[str] = None,
    ) -> Union[str, List[str]]:
        options = parse_options(plugin_option)
        count = int(options.get("n_variants", self.DEFAULT_COUNT))
        suffix = options.get("suffix", self.DEFAULT_SUFFIX)

        results = [text]
        for idx in range(1, count):
            if idx == 1:
                results.append(f"{text}{suffix}")
            else:
                results.append(f"{text}{suffix}-{idx}")
        return results
