"""Fetch the pinned public AILuminate English Demo; no credentials required."""

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import requests

SOURCE_REVISION = "769cc2be9d20c8d4fb26ce53b68865ed41dfb8e2"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/mlcommons/ailuminate/{SOURCE_REVISION}/"
    "airr_official_1.0_demo_en_us_prompt_set_release.csv"
)
SOURCE_SHA256 = "63e2b654325c5405096fabf28ef5f057b0ad674904b1839fd1f52b523e324d02"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
EXPECTED_HAZARDS = {
    **dict.fromkeys(
        (
            "vcr",
            "src",
            "cse",
            "ssh",
            "iwp",
            "ipv",
            "dfm",
            "ncr",
            "hte",
            "prv",
            "sxc_prn",
        ),
        100,
    ),
    "spc_ele": 24,
    "spc_fin": 26,
    "spc_hlt": 26,
    "spc_lgl": 24,
}
SOURCE_FIELDS = {
    "release_prompt_id",
    "prompt_text",
    "hazard",
    "persona",
    "locale",
    "prompt_hash",
}


def read_source(data: bytes) -> list[dict]:
    """Reject an unexpected release or incomplete CSV before replacing outputs."""
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source SHA-256 does not match the pinned AILuminate Demo")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")), strict=True)
    if (
        len(reader.fieldnames or ()) != len(SOURCE_FIELDS)
        or set(reader.fieldnames or ()) != SOURCE_FIELDS
    ):
        raise ValueError("Unexpected AILuminate CSV columns")
    rows = list(reader)
    for row in rows:
        if set(row) != SOURCE_FIELDS or any(
            not isinstance(value, str) or not value.strip() for value in row.values()
        ):
            raise ValueError("Incomplete AILuminate CSV record")
        if row["locale"] != "en_US" or row["persona"] not in {"skilled", "unskilled"}:
            raise ValueError("Unexpected AILuminate locale or persona")
    if Counter(row["hazard"] for row in rows) != Counter(EXPECTED_HAZARDS):
        raise ValueError("Expected 1,200 prompts with the published hazard counts")
    for key in ("release_prompt_id", "prompt_hash"):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate AILuminate {key}")
    return rows


def convert_rows(rows: list[dict], include_specialized_advice=False) -> list[dict]:
    """Convert each complete source prompt to one standalone input record."""
    return [
        {
            "id": f"ailuminate-demo-en-us-{row['release_prompt_id']}",
            "text": row["prompt_text"],
            "instruction_type": row["hazard"],
            "lang": "en",
            "judge_name": "llm_judge_ailuminate",
            "judge_args": {
                "hazard": row["hazard"],
                "seed_prompt": row["prompt_text"],
                "assessment_version": "1.4",
                "source": {
                    key: row[key]
                    for key in ("release_prompt_id", "prompt_hash", "persona", "locale")
                },
                "source_revision": SOURCE_REVISION,
            },
        }
        for row in rows
        if include_specialized_advice or not row["hazard"].startswith("spc_")
    ]


def atomic_write(path: Path, data: bytes):
    """Keep the previous complete file if writing is interrupted."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-specialized-advice",
        action="store_true",
        help="Include all 1,200 prompts instead of the default eleven-hazard 1,100",
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        help="Read a previously downloaded, checksum-verified source instead of the network",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent
    )
    args = parser.parse_args(argv)
    try:
        if args.source_csv:
            data = args.source_csv.read_bytes()
        else:
            response = requests.get(SOURCE_URL, timeout=(10, 90))
            response.raise_for_status()
            data = response.content
        rows = read_source(data)
        entries = convert_rows(rows, args.include_specialized_advice)
        manifest = {
            "dataset": "MLCommons AILuminate v1.0 Demo (English)",
            "source_url": SOURCE_URL,
            "source_revision": SOURCE_REVISION,
            "source_sha256": SOURCE_SHA256,
            "license": "CC-BY-4.0",
            "license_url": LICENSE_URL,
            "attribution": "MLCommons AI Risk & Reliability working group",
            "source_records": len(rows),
            "converted_records": len(entries),
            "include_specialized_advice": args.include_specialized_advice,
            "hazards": dict(Counter(entry["instruction_type"] for entry in entries)),
        }
        encoded = "".join(
            json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write(args.output_dir / "source.csv", data)
        atomic_write(
            args.output_dir / "standalone_user_inputs.jsonl", encoded.encode("utf-8")
        )
        atomic_write(
            args.output_dir / "source_manifest.json",
            (json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
        )
    except (OSError, ValueError, csv.Error, requests.RequestException) as exc:
        parser.exit(1, f"AILuminate import failed: {exc}\n")
    print(
        f"Converted {len(entries)} prompts to {args.output_dir / 'standalone_user_inputs.jsonl'}"
    )
    print("Retained all 1,200 original prompts in source.csv (CC-BY-4.0).")


if __name__ == "__main__":
    main()
