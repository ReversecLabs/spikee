from typing import List, Optional, Union

from spikee.templates.plugin import Plugin
from spikee.utilities.hinting import ModuleOptionsHint
from spikee.utilities.content import Text


class TestUpperPlugin(Plugin):
    def get_available_option_values(self) -> ModuleOptionsHint:
        return [], True

    def transform(
        self,
        content: Text,
        exclude_patterns: Optional[List[str]] = None,
        plugin_option: Optional[str] = None,
    ) -> Union[Text, List[Text]]:
        text = content.content
        return [Text(text.upper())]
