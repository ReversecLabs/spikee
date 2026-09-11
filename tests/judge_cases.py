"""Fixed, human-labelled responses to real bundled dataset entries.

Instruction seeds use the bundled direct/no-jailbreak wrapper, with the production
generator. Standalone seeds use its standalone path. No target or attacker runs.
"""

import json
from pathlib import Path

from spikee.generator import _process_permutation_worker, _process_standalone_worker
from spikee.utilities.enums import EntryType

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "spikee/data/workspace"
DATASETS = WORKSPACE / "datasets"


def source_row(dataset, filename, source_id):
    rows = [
        json.loads(line)
        for line in (DATASETS / dataset / filename).read_text().splitlines()
        if line.strip()
    ]
    matches = [row for row in rows if row["id"] == source_id]
    assert len(matches) == 1, (dataset, source_id, "missing or ambiguous source")
    return matches[0]


def materialize(case):
    source = source_row(case["dataset"], case["source_file"], case["source_id"])
    transforms = {"plugins": [(None, None)], "prefixes": [None], "suffixes": [None]}
    if case["source_file"] == "standalone_user_inputs.jsonl":
        entries = _process_standalone_worker({"attack": source, **transforms}, {})
        entry = entries[0].to_attack()
    else:
        wrapper = "seeds-harmful-instructions-only"
        entries = _process_permutation_worker(
            {
                **transforms,
                "base_doc": source_row(wrapper, "base_user_inputs.jsonl", "direct-01"),
                "jailbreak": source_row(wrapper, "jailbreaks.jsonl", "no-jailbreak-01"),
                "instruction": source,
                "positions": ["fixed"],
                "injection_delimiters": ["INJECTION_PAYLOAD"],
                "spotlighting_data_markers_list": ["none"],
                "match_languages": False,
            },
            {},
            None,
            [EntryType.DOCUMENT],
        )
        entry = entries[0].to_entry()
    # No bundled seed currently selects the objective judge. These cases explicitly
    # override the judge while retaining the original generated objective/content.
    if "judge_override" in case:
        entry["judge_name"] = case["judge_override"]
    return entry


def load_cases():
    return [
        json.loads(line)
        for line in (ROOT / "tests/fixtures/judge_cases.jsonl").read_text().splitlines()
        if line.strip()
    ]
