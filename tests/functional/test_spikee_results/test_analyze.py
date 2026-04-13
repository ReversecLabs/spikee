import pytest
from pathlib import Path

from spikee.utilities.files import read_jsonl_file
from spikee.utilities.results import (ResultProcessor)

from ..utils import spikee_generate_cli, spikee_test_cli, spikee_analyze_cli, create_judge_results

class TestResultProcessor:
    """Tests for the ResultProcessor class and its methods."""

    def get_processor(self, result_files, fp_check_file=None) -> ResultProcessor:
        if not isinstance(result_files, list):
            result_files = [result_files]

        combined_results = []
        for result_file in result_files:
            # Load the results data
            results = read_jsonl_file(result_file)

            for result in results:
                result["source_file"] = result_file  # Track source file for each entry

            combined_results.extend(results)

        return ResultProcessor(
            results=combined_results,
            result_file=result_files[0] if len(result_files) == 1 else "Combined",
            fp_check_file=fp_check_file,
        )
    
    @pytest.mark.parametrize("target_name", ["always_success", "always_success_legacy", "always_refuse", "always_refuse_legacy", "always_guardrail", "always_error"])
    def test_result_processor(self, run_spikee, workspace_dir, target_name):
        """Test that ResultProcessor correctly processes results and identifies false positives/negatives based on the judge mode."""
        dataset_path = spikee_generate_cli(run_spikee, workspace_dir)
        entries = read_jsonl_file(dataset_path)

        mode = "success" if "success" in target_name else "fail" if "refuse" in target_name else None

        results_file, _ = spikee_test_cli(
            run_spikee,
            workspace_dir,
            target=target_name,
            datasets=[dataset_path],
            additional_args=[
                "--no-auto-resume",
            ],
        )

        result_file = str(results_file[0] if isinstance(results_file, list) else results_file)
        processor = self.get_processor(result_file)

        overview = processor.generate_output(overview=True)
        assert overview is not None and len(overview) > 0, "Expected generate_output to produce output"

        output = processor.generate_output()
        assert output is not None and len(output) > 0, "Expected generate_output to produce output"

        assert processor.total_entries == len(entries), f"Expected total entries to match dataset entries ({len(entries)}), got {processor.total_entries}"
        assert processor.successful_groups == (len(entries) if mode is not None and mode == "success" else 0), f"Expected successful groups to match dataset entries when judge mode is success, got {processor.successful_groups}"
        assert processor.failed_groups == (len(entries) if mode is not None and mode == "fail" else 0), f"Expected failed groups to match dataset entries when judge mode is fail, got {processor.failed_groups}"
        assert processor.guardrail_groups == (len(entries) if target_name == "always_guardrail" else 0), f"Expected guardrail groups to match dataset entries when target is always_guardrail, got {processor.guardrail_groups}"
        assert processor.error_groups == (len(entries) if target_name == "always_error" or target_name == "always_guardrail" else 0), f"Expected error groups to match dataset entries when target is always_error, got {processor.error_groups}"

        assert processor.total_attempts == len(entries), f"Expected total attempts to match dataset entries ({len(entries)}), got {processor.total_attempts}"

        

    def test_combined_results(self, run_spikee, workspace_dir):
        """Test that analyze_results can process multiple results files and produce a combined analysis."""
        dataset_path = spikee_generate_cli(run_spikee, workspace_dir)
        entries = read_jsonl_file(dataset_path)

        results_files = []
        for target in ["always_success", "always_refuse"]:

            result_file, _ = spikee_test_cli(
                run_spikee,
                workspace_dir,
                target=target,
                datasets=[dataset_path],
                additional_args=[
                    "--no-auto-resume",
                ],
            )
            results_files.extend(str(f) for f in (result_file if isinstance(result_file, list) else [result_file]))

        assert len(results_files) == 2, f"Expected 2 results files for combined analysis, got {len(results_files)}"

        processor = self.get_processor(results_files)

        overview = processor.generate_output(overview=True)
        assert overview is not None and len(overview) > 0, "Expected generate_output to produce output"

        output = processor.generate_output()
        assert output is not None and len(output) > 0, "Expected generate_output to produce output"

        assert processor.total_entries == len(entries) * 2, f"Expected total entries to match twice the dataset entries ({len(entries) * 2}), got {processor.total_entries}"

        assert processor.successful_groups == len(entries), f"Expected successful groups to match dataset entries when judge mode is success, got {processor.successful_groups}"
        assert processor.failed_groups == len(entries), f"Expected failed groups to match dataset entries when judge mode is fail, got {processor.failed_groups}"
        assert processor.guardrail_groups == 0, f"Expected guardrail groups to match dataset entries when target is always_guardrail, got {processor.guardrail_groups}"
        assert processor.error_groups == 0, f"Expected error groups to match dataset entries when target is always_error, got {processor.error_groups}"

        assert processor.total_attempts == len(entries) * 2, f"Expected total attempts to match dataset entries ({len(entries) * 2}), got {processor.total_attempts}"

        
    

class TestAnalyzeResults:
    """Tests for the analyze_results function and its integration with the CLI."""

    @pytest.mark.parametrize("target_name", ["always_success", "always_success_legacy"])
    @pytest.mark.parametrize("judge_variant", ["test_judge", "test_judge_legacy"])
    def test_analyze_result_file(self, run_spikee, workspace_dir, target_name, judge_variant):
        """Test that the analyze command produces expected output based on the judge mode."""
        results_file = create_judge_results(run_spikee, workspace_dir, target_name, judge_variant)

        output = spikee_analyze_cli(
            run_spikee,
            workspace_dir,
            result_files=[results_file[0] if isinstance(results_file, list) else results_file],
        )

        assert "General Statistics" in output
        assert "Total Unique Entries" in output
        assert "Successful Attacks: 0" in output
        assert "Attack Success Rate" in output

    def test_analyze_result_overview(self, run_spikee, workspace_dir):
        """Test that the analyze command produces expected output based on the judge mode."""
        results_file = create_judge_results(run_spikee, workspace_dir, "always_success", "test_judge")

        output = spikee_analyze_cli(
            run_spikee,
            workspace_dir,
            result_files=[results_file[0] if isinstance(results_file, list) else results_file],
            additional_args=["--overview"],
        )

        assert "=== Breakdown by" not in output

    def test_analyze_results_folder(self, run_spikee, workspace_dir):
        """Test that the analyze command produces expected output based on the judge mode."""

        results_files = []
        for target in ["always_success", "always_refuse"]:
            results_files.append(create_judge_results(run_spikee, workspace_dir, target, "test_judge"))

        output = spikee_analyze_cli(
            run_spikee,
            workspace_dir,
            result_files=[Path(workspace_dir) / "results" / ""],
        )

        assert "General Statistics" in output
        assert "Total Unique Entries" in output
        assert "Successful Attacks: 0" in output
        assert "Attack Success Rate" in output
        assert "Combined" not in output

        output = spikee_analyze_cli(
            run_spikee,
            workspace_dir,
            result_files=results_files,
        )

        assert "General Statistics" in output
        assert "Total Unique Entries" in output
        assert "Successful Attacks: 0" in output
        assert "Attack Success Rate" in output
        assert "Combined" not in output

        output = spikee_analyze_cli(
            run_spikee,
            workspace_dir,
            result_files=results_files,
            additional_args=["--combine"],
        )

        assert "Combined" in output