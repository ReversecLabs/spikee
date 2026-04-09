import shutil
import re

from ..utils import spikee_generate_cli


def test_progress_bar_shows_correct_total(run_spikee, workspace_dir):
    # 1. Generate a dataset (standard small dataset)
    dataset_path = spikee_generate_cli(run_spikee, workspace_dir)

    # 2. Rename to a very long filename to distinguish string length from entry count.
    # We use a filename > 200 chars.
    # The 'buggy' code uses len(path), which will be len(abs_path) or len(rel_path).
    # Either way, if filename is 200 chars, the path string len is >= 200.
    # The entry count for default generate is typically small (< 50).

    long_name = "a" * 200 + ".jsonl"
    new_path = dataset_path.parent / long_name
    shutil.move(dataset_path, new_path)

    dataset_rel = new_path.relative_to(workspace_dir)

    # 3. Run test command
    # We use "always_success" target to ensure it runs quickly
    result = run_spikee(["test", "--target", "always_success", "--dataset", str(dataset_rel)], cwd=workspace_dir)

    # 4. Check stderr for progress bar totals
    # Tqdm progress bars output patterns like " 5/20 " or " 5/200 " or "100%|...| 5/20"
    # We look for the "total" part: the number after the slash.

    stderr = result.stderr

    # Find all occurrences of "/N" where N is a number
    # using regex `/\d+`
    # (Note: tqdm might use space after slash? Usually not: "5/20")

    totals = [int(m.group(1)) for m in re.finditer(r"/(\d+)", stderr)]

    print(f"DEBUG: Found totals in stderr: {totals}")

    # If the bug is present, one of the totals will be the length of the path string (>= 200).
    # If the bug is fixed, all totals should be the entry count (small, < 100).

    large_totals = [t for t in totals if t >= 200]

    if large_totals:
        pytest_fail_msg = (
            f"Found suspicious progress bar total(s) {large_totals} in stderr. "
            f"This likely indicates the file path length was used as the total. "
            f"Stderr snippet:\n{stderr[:1000]}..."
        )
        assert not large_totals, pytest_fail_msg
