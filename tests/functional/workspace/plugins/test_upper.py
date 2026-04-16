from typing import List, Optional, Union

from spikee.templates.plugin import Plugin
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint
from spikee.utilities.content import Text


class TestUpperPlugin(Plugin):
    def get_description(self) -> ModuleDescriptionHint:
        return [], "Test plugin for converting content to uppercase."

    def get_available_option_values(self) -> ModuleOptionsHint:
        return [], False

    def transform(
        self,
        content: Text,
        exclude_patterns: Optional[List[str]] = None,
        plugin_option: Optional[str] = None,
    ) -> Union[Text, List[Text]]:
        text = content.content
        return [Text(text.upper())]
