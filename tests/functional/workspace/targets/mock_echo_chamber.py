"""Deterministic target implementing Echo Chamber's seed-generation contract."""

import json

from spikee.templates.multi_target import MultiTarget
from spikee.utilities.enums import Turn


class MockEchoChamberTarget(MultiTarget):
    def __init__(self):
        super().__init__(turn_types=[Turn.MULTI], backtrack=True)

    def process_input(
        self,
        input_text,
        system_message=None,
        target_options=None,
        spikee_session_id=None,
        backtrack=False,
    ):
        if '"keywords"' in input_text:
            return json.dumps(
                {
                    "sentences": [
                        "Paris is the capital of France.",
                        "The French capital is home to the Eiffel Tower.",
                        "Paris sits on the River Seine.",
                    ]
                }
            )
        return "Paris is the capital of France."
