from typing import List, Optional, Tuple, Union

from spikee.templates.plugin import Plugin


class TestRepeatPlugin(Plugin):
    DEFAULT_SUFFIX = "-repeat"
    DEFAULT_COUNT = 2

    def get_available_option_values(self) -> List[str]:
        return [
            "n_variants=2",
            "n_variants=<int>,suffix=<suffix>",
        ]

    def _parse_options(self, option_string: Optional[str]) -> Tuple[int, str]:
        count = self.DEFAULT_COUNT
        suffix = self.DEFAULT_SUFFIX
        if not option_string:
            return count, suffix

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

        suffix = options.get("suffix", suffix) or suffix
        return count, suffix

    def transform(
        self,
        text: str,
        exclude_patterns: Optional[List[str]] = None,
        plugin_option: Optional[str] = None,
    ) -> Union[str, List[str]]:
        count, suffix = self._parse_options(plugin_option)

        results = [text]
        for idx in range(1, count):
            if idx == 1:
                results.append(f"{text}{suffix}")
            else:
                results.append(f"{text}{suffix}-{idx}")
        return results
