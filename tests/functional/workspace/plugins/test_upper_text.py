from typing import List, Optional, Union

from spikee.templates.plugin import Plugin
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint


class TestUpperPlugin(Plugin):
    def get_description(self) -> ModuleDescriptionHint:
        return [], "Test plugin for converting text to uppercase."

    def get_available_option_values(self) -> ModuleOptionsHint:
        return [], False

    def transform(
        self,
        text: str,
        exclude_patterns: Optional[List[str]] = None,
        plugin_option: Optional[str] = None,
    ) -> Union[str, List[str]]:
        return [text.upper()]
