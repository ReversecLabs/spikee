from pathlib import Path
import time

from spikee.utilities.files import read_jsonl_file, write_jsonl_file
from ..utils import spikee_test_cli, spikee_generate_cli


class TestDatasetArguments:
    """Test cases for --dataset, --dataset-folder, and combinations."""

    def test_single_dataset(self, run_spikee, workspace_dir):
        """Test --dataset with a single file produces one results file with all entries tested."""
        dataset_path = spikee_generate_cli(run_spikee, workspace_dir)

        results_files, _ = spikee_test_cli(
            run_spikee,
            workspace_dir,
            target="always_success",
            datasets=[dataset_path],
        )

        assert len(results_files) == 1, \
            f"Expected 1 results file for a single dataset, got {len(results_files)}"

        results = read_jsonl_file(results_files[0])
        dataset = read_jsonl_file(dataset_path)

        assert len(results) == len(dataset), \
            f"Expected {len(dataset)} result entries, got {len(results)}"

        # always_success returns the canary string — all entries should succeed
        assert all(r["success"] for r in results), \
            "Expected all entries to succeed with always_success target"

        # Every result should have expected fields
        for r in results:
            assert "id" in r, "Result missing 'id' field"
            assert "long_id" in r, "Result missing 'long_id' field"
            assert "success" in r, "Result missing 'success' field"
            assert "response" in r, "Result missing 'response' field"

    def test_dataset_folder(self, run_spikee, workspace_dir):
        """Test --dataset-folder runs testing against all JSONL files in the folder.

        Generates two datasets into the datasets/ folder, then passes the folder.
        Each dataset gets its own results file, so we expect 2 results files.
        """
        spikee_generate_cli(
            run_spikee, workspace_dir, additional_args=["--languages", "en"]
        )
        spikee_generate_cli(
            run_spikee, workspace_dir, additional_args=["--languages", "it"]
        )

        datasets_folder = workspace_dir / "datasets"

        results_files, _ = spikee_test_cli(
            run_spikee,
            workspace_dir,
            target="always_refuse",
            datasets=[datasets_folder],
            additional_args=["--no-auto-resume"],
        )

        assert len(results_files) >= 2, \
            f"Expected at least 2 results files (one per dataset), got {len(results_files)}"

        # All results should report failure
        for rf in results_files:
            results = read_jsonl_file(rf)
            assert len(results) > 0, f"Results file {rf.name} is empty"
            assert all(not r["success"] for r in results), \
                f"Expected all entries to fail in {rf.name}"

    def test_multiple_datasets_combined(self, run_spikee, workspace_dir):
        """Test passing multiple --dataset flags combines entries from both files
        into separate results files (one per dataset).
        """
        dataset_en = spikee_generate_cli(
            run_spikee, workspace_dir, additional_args=["--languages", "en"]
        )
        dataset_it = spikee_generate_cli(
            run_spikee, workspace_dir, additional_args=["--languages", "it"]
        )

        entries_en = read_jsonl_file(dataset_en)
        entries_it = read_jsonl_file(dataset_it)

        results_files, _ = spikee_test_cli(
            run_spikee,
            workspace_dir,
            target="always_success",
            datasets=[dataset_en, dataset_it],
            additional_args=["--no-auto-resume"],
        )

        assert len(results_files) == 2, \
            f"Expected 2 results files (one per dataset), got {len(results_files)}"

        total_results = sum(len(read_jsonl_file(rf)) for rf in results_files)
        expected_total = len(entries_en) + len(entries_it)
        assert total_results == expected_total, \
            f"Expected {expected_total} total results across both files, got {total_results}"


class TestResume:
    """Test cases for --result-file, --auto-resume, and --no-auto-resume"""

    def create_partial_results(self, dataset_path: Path, num_entries: int, target_name: str, workspace_dir: Path) -> Path:
        """Helper function to create a partial results file for a given dataset."""
        entries = read_jsonl_file(dataset_path)
        ds_name = dataset_path.stem

        completed_entries = []
        for entry in entries[:num_entries]:
            completed_entries.append({
                "id": entry["id"],
                "long_id": entry.get("long_id", entry["id"]),
                "success": True,
                "response": "canary response",
            })

        results_dir = workspace_dir / "results"
        results_dir.mkdir(exist_ok=True)

        resume_file = results_dir / f"results_{target_name}_{ds_name}_{int(time.time())}.jsonl"
        write_jsonl_file(resume_file, completed_entries)

        return resume_file

    def test_single_dataset_resume(self, run_spikee, workspace_dir):
        """Test that --auto-resume correctly resumes from a partial results file for a single dataset."""
        # 1. Generate a dataset
        dataset_path = spikee_generate_cli(run_spikee, workspace_dir, additional_args=["--tag", "resume_test"])
        entries = read_jsonl_file(dataset_path)

        # 2. Create partial results file with first 2 entries completed
        resume_file = self.create_partial_results(dataset_path, num_entries=2, target_name="always_success", workspace_dir=workspace_dir)

        # 3. Run test with --auto-resume
        results_files, result = spikee_test_cli(
            run_spikee,
            workspace_dir,
            target="always_success",
            datasets=[dataset_path],
            additional_args=["--auto-resume"],
        )

        # 4. Assertions
        stdout = result.stdout
        results = read_jsonl_file(results_files[0])

        assert len(results_files) == 1, f"Expected 1 results file after resuming, got {len(results_files)}"
        assert f"[Auto-Resume] Using latest: {resume_file.name}" in stdout
        assert len(results) == len(entries), f"Expected all {len(entries)} entries to be processed after resuming, got {len(results)}"

        # Check that the resumed results file contains the canary response for the first 2 entries
        for r in results[:2]:
            assert r["success"], "Expected resumed entries to be marked as success"
            assert r["response"] == "canary response", "Expected resumed entries to have the canary response"

    def test_single_dataset_resume_file(self, run_spikee, workspace_dir):
        """Test that --result-file can be used to specify a resume file for a single dataset."""
        # 1. Generate a dataset
        dataset_path = spikee_generate_cli(run_spikee, workspace_dir, additional_args=["--tag", "resume_file_test"])
        entries = read_jsonl_file(dataset_path)

        # 2. Create partial results file with first 2 entries completed
        resume_file = self.create_partial_results(dataset_path, num_entries=2, target_name="always_success", workspace_dir=workspace_dir)

        # 3. Run test with --result-file pointing to the resume file
        results_files, result = spikee_test_cli(
            run_spikee,
            workspace_dir,
            target="always_success",
            datasets=[dataset_path],
            additional_args=["--resume-file", str(resume_file)],
        )

        # 4. Assertions
        results = read_jsonl_file(results_files[0])

        assert len(results_files) == 1, f"Expected 1 results file after resuming, got {len(results_files)}"
        assert len(results) == len(entries), f"Expected all {len(entries)} entries to be processed after resuming, got {len(results)}"

        # Check that the resumed results file contains the canary response for the first 2 entries
        for r in results[:2]:
            assert r["success"], "Expected resumed entries to be marked as success"
            assert r["response"] == "canary response", "Expected resumed entries to have the canary response"

    def test_single_dataset_no_resume(self, run_spikee, workspace_dir):
        """Test that --no-auto-resume correctly ignores existing results files and starts fresh."""
        # 1. Generate a dataset
        dataset_path = spikee_generate_cli(run_spikee, workspace_dir, additional_args=["--tag", "no_resume_test"])
        entries = read_jsonl_file(dataset_path)

        # 2. Create partial results file with first 2 entries completed
        resume_file = self.create_partial_results(dataset_path, num_entries=2, target_name="always_success", workspace_dir=workspace_dir)

        # 3. Run test with --no-auto-resume
        results_files, result = spikee_test_cli(
            run_spikee,
            workspace_dir,
            target="always_success",
            datasets=[dataset_path],
            additional_args=["--no-auto-resume"],
        )

        # 4. Assertions
        stdout = result.stdout
        results = read_jsonl_file(results_files[0])

        assert len(results_files) == 1, f"Expected 1 results file after running with no auto-resume, got {len(results_files)}"
        assert f"[Auto-Resume] Using specified resume file: {resume_file.name}" not in stdout
        assert len(results) == len(entries), f"Expected all {len(entries)} entries to be processed when not resuming, got {len(results)}"

        # Check that the new results file does NOT contain the canary response for the first 2 entries (since it should have started fresh)
        for r in results[:2]:
            assert r["response"] != "canary response", "Expected new run to not use canary response from resume file"

    def test_multiple_datasets_independent_resume(self, run_spikee, workspace_dir):
        """Test that when running with multiple datasets, --auto-resume correctly resumes each dataset independently from its own resume file."""
        # 1. Generate 2 datasets
        dataset_path_a = spikee_generate_cli(run_spikee, workspace_dir, additional_args=["--tag", "resume_multi_a", "--languages", "en"])
        dataset_path_b = spikee_generate_cli(run_spikee, workspace_dir, additional_args=["--tag", "resume_multi_b", "--languages", "en"])

        entries_a = read_jsonl_file(dataset_path_a)
        entries_b = read_jsonl_file(dataset_path_b)

        # 2. Create partial results for Dataset A
        resume_file_a = self.create_partial_results(dataset_path_a, num_entries=2, target_name="always_success", workspace_dir=workspace_dir)
        # Dataset B has NO results.

        # 3. Run test for BOTH datasets with auto-resume
        results_files, result = spikee_test_cli(
            run_spikee,
            workspace_dir,
            target="always_success",
            datasets=[dataset_path_a, dataset_path_b],
            additional_args=["--auto-resume"],
        )

        # 4. Assertions
        stdout = result.stdout
        results_a = read_jsonl_file(results_files[0])
        results_b = read_jsonl_file(results_files[1])

        assert len(results_files) == 2, f"Expected 2 results files after resuming multiple datasets, got {len(results_files)}"
        assert f"[Auto-Resume] Using latest: {resume_file_a.name}" in stdout
        assert len(results_a) == len(entries_a), f"Expected all {len(entries_a)} entries to be processed for Dataset A after resuming, got {len(results_a)}"
        assert len(results_b) == len(entries_b), f"Expected all {len(entries_b)} entries to be processed for Dataset B, got {len(results_b)}"

        # Check that the resumed results file for Dataset A contains the canary response for the first 2 entries
        potential_canary = results_a[:2] + results_b[:2]
        canary_count = sum(1 for r in potential_canary if r["response"] == "canary response")
        assert canary_count == 2, f"Expected exactly 2 entries with canary response across both datasets, got {canary_count}"

    def test_dataset_auto_resume_picks_latest_candidate(self, run_spikee, workspace_dir):
        """Test that when multiple resume candidates are present, --auto-resume picks the one with the latest timestamp."""
        dataset_path = spikee_generate_cli(run_spikee, workspace_dir, additional_args=["--tag", "resume_test"])
        entries = read_jsonl_file(dataset_path)

        self.create_partial_results(dataset_path, num_entries=2, target_name="always_refuse", workspace_dir=workspace_dir)
        time.sleep(1)  # Ensure the second resume file has a later timestamp
        new_resume_file = self.create_partial_results(dataset_path, num_entries=4, target_name="always_success", workspace_dir=workspace_dir)

        results_files, result = spikee_test_cli(
            run_spikee,
            workspace_dir,
            target="always_success",
            datasets=[dataset_path],
            additional_args=["--auto-resume"],
        )

        stdout = result.stdout
        results = read_jsonl_file(results_files[0])

        assert len(results_files) == 1, f"Expected 1 results file after resuming, got {len(results_files)}"
        assert f"[Auto-Resume] Using latest: {new_resume_file.name}" in stdout
        assert len(results) == len(entries), f"Expected all {len(entries)} entries to be processed after resuming, got {len(results)}"

        for r in results:
            assert r["success"], "Expected resumed entries to be marked as success based on the latest resume file"

    def test_dataset_skips_complete_dataset_auto_resume(self, run_spikee, workspace_dir):
        """Test that when a resume file is present with all entries marked as complete, --auto-resume skips it and starts fresh."""
        dataset_path = spikee_generate_cli(run_spikee, workspace_dir, additional_args=["--tag", "complete_resume_test"])
        entries = read_jsonl_file(dataset_path)

        resume_file = self.create_partial_results(dataset_path, num_entries=len(entries), target_name="always_refuse", workspace_dir=workspace_dir)

        results_files, result = spikee_test_cli(
            run_spikee,
            workspace_dir,
            target="always_success",
            datasets=[dataset_path],
            additional_args=["--auto-resume"],
        )

        stdout = result.stdout
        results = read_jsonl_file(results_files[0])

        assert len(results_files) == 1, f"Expected 1 results file after running with auto-resume on complete dataset, got {len(results_files)}"
        assert f"[Auto-Resume] Using latest: {resume_file.name}" not in stdout
        assert len(results) == len(entries), f"Expected all {len(entries)} entries to be processed when auto-resume skips complete dataset, got {len(results)}"

        # Check that the new results file does NOT contain the canary response for any entries (since it should have started fresh)
        for r in results:
            assert r["response"] != "canary response", "Expected new run to not use canary response from complete resume file"
