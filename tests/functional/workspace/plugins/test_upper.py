from typing import List, Optional, Union

from spikee.templates.plugin import Plugin


class TestUpperPlugin(Plugin):
    def get_available_option_values(self) -> List[str]:
        return []

    def transform(
        self,
        text: str,
        exclude_patterns: Optional[List[str]] = None,
        plugin_option: Optional[str] = None,
    ) -> Union[str, List[str]]:
        return [text.upper()]
