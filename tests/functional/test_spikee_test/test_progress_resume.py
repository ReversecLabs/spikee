import re

from spikee.utilities.files import read_jsonl_file, write_jsonl_file
from ..utils import spikee_generate_cli


def test_progress_bar_resume_total_correctness(run_spikee, workspace_dir):
    # 1. Generate a dataset with known size (e.g. 10 entries)
    # We can control size by editing the file or assuming default.
    # Let's generate and then truncate/expand if needed, or just count.
    dataset_path = spikee_generate_cli(run_spikee, workspace_dir)
    entries = read_jsonl_file(dataset_path)
    full_count = len(entries)

    # We need at least some entries to split.
    assert full_count > 2, "Dataset too small for resume test"

    # 2. Create a partial results file to simulate a previous run.
    # Let's say we processed half.
    processed_count = int(full_count / 2)
    completed_entries = []
    for i in range(processed_count):
        # Create a dummy result entry
        res = {
            "id": entries[i]["id"],
            "long_id": entries[i]["long_id"],
            "success": False,
            "attack_name": "None",
            "attempts": 1,
        }
        completed_entries.append(res)

    resume_file = workspace_dir / "results_resume.jsonl"
    write_jsonl_file(resume_file, completed_entries)

    dataset_rel = dataset_path.relative_to(workspace_dir)
    resume_rel = resume_file.relative_to(workspace_dir)

    # 3. Run test command with --resume-file
    # Target "always_success"
    result = run_spikee(
        ["test", "--target", "always_success", "--dataset", str(dataset_rel), "--resume-file", str(resume_rel)],
        cwd=workspace_dir,
    )

    # 4. Check stderr for progress bar totals
    # We expect the total to be `full_count`.
    # Current bug might show `full_count - processed_count`.

    stderr = result.stderr
    print(f"DEBUG: Stderr:\n{stderr}")

    # Parse stderr for specific lines
    processing_bar_lines = [
        line for line in stderr.splitlines() if "Processing entries" in line
    ]
    print(f"DEBUG: Processing bar lines: {processing_bar_lines}")

    # Extract totals from these lines
    # Pattern: "... N/M" e.g. " 15/458"
    proc_totals = []
    for line in processing_bar_lines:
        match = re.search(r"/(\d+)", line)
        if match:
            proc_totals.append(int(match.group(1)))

    print(f"DEBUG: Processing bar totals: {proc_totals}")

    # We want at least one update showing the FULL total
    has_full_count = any(t == full_count for t in proc_totals)

    # If bug is present, we see (full_count - processed_count)
    buggy_count = full_count - processed_count
    has_buggy_count = any(t == buggy_count for t in proc_totals)

    if has_buggy_count and not has_full_count:
        assert False, (
            f"Found buggy total {buggy_count} in Processing Entries bar. Expected {full_count}."
        )

    assert has_full_count, (
        f"Did not find expected total {full_count} in Processing Entries bar. Found: {proc_totals}"
    )
