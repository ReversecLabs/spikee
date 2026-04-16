from typing import List, Optional, Union

from spikee.templates.plugin import Plugin
from spikee.utilities.modules import parse_options
from spikee.utilities.llm import get_llm
from spikee.utilities.llm_message import HumanMessage

class TestInference(Plugin):
    def get_available_option_values(self) -> List[str]:
        return []

    def transform(
        self,
        text: str,
        exclude_patterns: Optional[List[str]] = None,
        plugin_option: Optional[str] = None,
    ) -> Union[str, List[str]]:
        options = parse_options(plugin_option)

        model = options.get("model", "openai/gpt-4o")

        llm = get_llm(model, max_tokens=100)

        messages = [
            HumanMessage(content=f"Echo the following text verbatim: {text}")
        ]

        response = llm.invoke(messages).content

        return [response]
