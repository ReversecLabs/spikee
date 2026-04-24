"""Plugin that uppercases content while preserving type."""
from typing import Optional

from spikee.templates.basic_plugin import BasicPlugin
from spikee.utilities.hinting import ModuleOptionsHint, Content, get_content, get_content_type, content_factory


class UppercaseContentPlugin(BasicPlugin):
    """Uppercase transformation that preserves content type."""

    def get_available_option_values(self) -> ModuleOptionsHint:
        return [], False

    def plugin_transform(self, text: Content, plugin_option: Optional[str] = None) -> Content:
        """Transform text to uppercase."""
        return content_factory(get_content(text).upper(), get_content_type(text))
