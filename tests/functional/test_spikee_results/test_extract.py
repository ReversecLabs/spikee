import subprocess
import pytest

from spikee.utilities.files import read_jsonl_file
from spikee.utilities.results import extract_entries, extract_search, generate_query
from ..utils import spikee_generate_cli, spikee_test_cli, spikee_extract_cli


# ---------------------------------------------------------------------------
# TestExtractSearch
# ---------------------------------------------------------------------------

class TestExtractSearch:
    def test_plain_match(self):
        assert extract_search({"response": "hello"}, "hello", "response") is True

    def test_plain_no_match(self):
        assert extract_search({"response": "hello"}, "world", "response") is False

    def test_invert_query_match(self):
        # "!hello" in an entry that contains "hello" → False (inverted)
        assert extract_search({"response": "hello"}, "!hello", "response") is False

    def test_invert_query_no_match(self):
        # "!world" in an entry that does not contain "world" → True (inverted)
        assert extract_search({"response": "hello"}, "!world", "response") is True

    def test_no_field_searches_full_entry(self):
        # No field → str(entry) is searched
        assert extract_search({"response": "hello"}, "hello") is True

    def test_missing_field_returns_false(self):
        assert extract_search({"response": "x"}, "hello", "nonexistent") is False

    def test_missing_field_inverted_returns_true(self):
        # field "!nonexistent" → field absent → return f_invert=True
        assert extract_search({"response": "x"}, "hello", "!nonexistent") is True

    def test_invert_field_present_match(self):
        # field "!a" but field "a" IS present → f_invert is ignored, normal search runs.
        # "hello" in str("hello") → True
        assert extract_search({"a": "hello"}, "hello", "!a") is True

    def test_invert_field_present_no_match(self):
        # field "!a" IS present, query not in value → False (f_invert still ignored)
        assert extract_search({"a": "hello"}, "world", "!a") is False


# ---------------------------------------------------------------------------
# TestExtractEntries
# ---------------------------------------------------------------------------

class TestExtractEntries:
    def test_success_true(self):
        assert extract_entries({"success": True}, "success") is True

    def test_success_false(self):
        assert extract_entries({"success": False}, "success") is False

    def test_failure_false(self):
        assert extract_entries({"success": False}, "failure") is True

    def test_failure_true(self):
        assert extract_entries({"success": True}, "failure") is False

    def test_error_with_message(self):
        assert extract_entries({"error": "timeout"}, "error") is True

    def test_error_no_response_message(self):
        assert extract_entries({"error": "No response received"}, "error") is False

    def test_error_none(self):
        assert extract_entries({"error": None}, "error") is False

    def test_guardrail_true(self):
        assert extract_entries({"guardrail": True}, "guardrail") is True

    def test_guardrail_false(self):
        assert extract_entries({"guardrail": False}, "guardrail") is False

    def test_no_guardrail_false(self):
        assert extract_entries({"guardrail": False}, "no-guardrail") is True

    def test_no_guardrail_true(self):
        assert extract_entries({"guardrail": True}, "no-guardrail") is False

    def test_custom_plain_match(self):
        assert extract_entries({"response": "flag"}, "custom", [["flag"]]) is True

    def test_custom_field_match(self):
        assert extract_entries({"response": "flag"}, "custom", [["flag", "response"]]) is True

    def test_custom_field_no_match(self):
        assert extract_entries({"response": "clean"}, "custom", [["flag", "response"]]) is False

    def test_custom_multiple_conditions_all_match(self):
        entry = {"response": "flag", "success": "True"}
        assert extract_entries(entry, "custom", [["flag", "response"], ["True", "success"]]) is True

    def test_custom_multiple_conditions_partial_match(self):
        entry = {"response": "flag"}
        assert extract_entries(entry, "custom", [["flag", "response"], ["other", "response"]]) is False

    def test_custom_inverted_query(self):
        assert extract_entries({"response": "clean"}, "custom", [["!flag", "response"]]) is True

    # -- multi-condition: all-inverted --

    def test_custom_all_inverted_both_absent_match(self):
        # Both inverted conditions pass — neither term is present
        entry = {"r": "clean"}
        assert extract_entries(entry, "custom", [["!flag", "r"], ["!poison", "r"]]) is True

    def test_custom_all_inverted_one_present_fail(self):
        # First inverted condition fails because "flag" IS present
        entry = {"r": "flag clean"}
        assert extract_entries(entry, "custom", [["!flag", "r"], ["!poison", "r"]]) is False

    # -- multi-condition: mixed normal + inverted --

    def test_custom_mixed_normal_and_inverted_match(self):
        # Normal "flag" matches AND inverted "poison" is absent
        entry = {"r": "flag", "s": "ok"}
        assert extract_entries(entry, "custom", [["flag", "r"], ["!poison", "s"]]) is True

    def test_custom_mixed_normal_and_inverted_fail(self):
        # Normal passes but inverted fails — "poison" IS present
        entry = {"r": "flag", "s": "poison"}
        assert extract_entries(entry, "custom", [["flag", "r"], ["!poison", "s"]]) is False

    # -- three-condition chains --

    def test_custom_three_conditions_all_match(self):
        entry = {"a": "x", "b": "y", "c": "z"}
        assert extract_entries(entry, "custom", [["x", "a"], ["y", "b"], ["z", "c"]]) is True

    def test_custom_three_conditions_middle_fails(self):
        entry = {"a": "x", "b": "y", "c": "z"}
        assert extract_entries(entry, "custom", [["x", "a"], ["NOPE", "b"], ["z", "c"]]) is False

    def test_custom_three_conditions_last_fails(self):
        entry = {"a": "x", "b": "y", "c": "z"}
        assert extract_entries(entry, "custom", [["x", "a"], ["y", "b"], ["NOPE", "c"]]) is False

    # -- multi-condition: plain (no-field) --

    def test_custom_plain_multi_both_match(self):
        # Two plain conditions both present in str(entry)
        entry = {"r": "alpha beta"}
        assert extract_entries(entry, "custom", [["alpha"], ["beta"]]) is True

    def test_custom_plain_multi_one_misses(self):
        entry = {"r": "alpha beta"}
        assert extract_entries(entry, "custom", [["alpha"], ["gamma"]]) is False

    # -- generate_query round-trips --

    def test_custom_inverted_field_round_trip_match(self):
        # "!response:flag": field absent (value is None) → f_invert=True → True
        query = generate_query("custom", ["!response:flag"])
        assert extract_entries({"response": None}, "custom", query) is True

    def test_custom_value_with_colon_round_trip(self):
        # Colon in value is preserved end-to-end
        query = generate_query("custom", ["url:http://x.com"])
        assert extract_entries({"url": "http://x.com"}, "custom", query) is True


# ---------------------------------------------------------------------------
# TestGenerateQuery
# ---------------------------------------------------------------------------

class TestGenerateQuery:
    def test_non_custom_category_returns_empty(self):
        assert generate_query("success") == []

    def test_non_custom_failure_returns_empty(self):
        assert generate_query("failure") == []

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError):
            generate_query("invalid")

    def test_custom_without_search_raises(self):
        with pytest.raises(ValueError):
            generate_query("custom", None)

    def test_custom_plain_string(self):
        assert generate_query("custom", ["flag"]) == [["flag"]]

    def test_custom_field_string(self):
        # "response:flag" → split(":", 1) → ["response", "flag"] → reversed → ["flag", "response"]
        assert generate_query("custom", ["response:flag"]) == [["flag", "response"]]

    def test_custom_multiple_strings(self):
        result = generate_query("custom", ["response:flag", "success:True"])
        assert result == [["flag", "response"], ["True", "success"]]

    def test_custom_inverted_query_preserved(self):
        # "!" is not consumed by generate_query, only by extract_search
        assert generate_query("custom", ["!flag"]) == [["!flag"]]

    def test_custom_inverted_field_preserved(self):
        # "!response:flag" → split → ["!response", "flag"] → reversed → ["flag", "!response"]
        # The "!" rides on the field token and is consumed later by extract_search
        assert generate_query("custom", ["!response:flag"]) == [["flag", "!response"]]

    def test_custom_value_with_colon(self):
        # split(":", 1) means only the first colon is consumed; rest of value is preserved
        assert generate_query("custom", ["url:http://x.com"]) == [["http://x.com", "url"]]


# ---------------------------------------------------------------------------
# TestExtractResultsCLI
# ---------------------------------------------------------------------------

class TestExtractResultsCLI:
    def _run_test(self, run_spikee, workspace_dir, target):
        """Generate a dataset, run a test with the given target, return (results_files, entries)."""
        dataset_path = spikee_generate_cli(run_spikee, workspace_dir)
        entries = read_jsonl_file(dataset_path)
        results_files, _ = spikee_test_cli(
            run_spikee, workspace_dir, target=target, datasets=[dataset_path],
            additional_args=["--no-auto-resume"],
        )
        return results_files, entries

    def test_extract_success(self, run_spikee, workspace_dir):
        results_files, entries = self._run_test(run_spikee, workspace_dir, "always_success")
        extract_files, _ = spikee_extract_cli(run_spikee, workspace_dir, result_files=results_files, category="success")

        assert len(extract_files) == 1
        extracted = read_jsonl_file(str(extract_files[0]))
        assert len(extracted) == len(entries)
        assert all(e["success"] is True for e in extracted)

    def test_extract_failure(self, run_spikee, workspace_dir):
        results_files, entries = self._run_test(run_spikee, workspace_dir, "always_refuse")
        extract_files, _ = spikee_extract_cli(run_spikee, workspace_dir, result_files=results_files, category="failure")

        assert len(extract_files) == 1
        extracted = read_jsonl_file(str(extract_files[0]))
        assert len(extracted) == len(entries)
        assert all(e["success"] is False for e in extracted)

    def test_extract_success_from_mixed(self, run_spikee, workspace_dir):
        results_files, entries = self._run_test(run_spikee, workspace_dir, "partial_success")
        extract_files, _ = spikee_extract_cli(run_spikee, workspace_dir, result_files=results_files, category="success")

        assert len(extract_files) == 1
        extracted = read_jsonl_file(str(extract_files[0]))
        assert 0 < len(extracted) < len(entries)
        assert all(e["success"] is True for e in extracted)

    def test_extract_guardrail(self, run_spikee, workspace_dir):
        results_files, entries = self._run_test(run_spikee, workspace_dir, "always_guardrail")
        extract_files, _ = spikee_extract_cli(run_spikee, workspace_dir, result_files=results_files, category="guardrail")

        assert len(extract_files) == 1
        extracted = read_jsonl_file(str(extract_files[0]))
        assert len(extracted) == len(entries)
        assert all(e["guardrail"] is True for e in extracted)

    def test_extract_no_guardrail(self, run_spikee, workspace_dir):
        results_files, _ = self._run_test(run_spikee, workspace_dir, "always_guardrail")
        extract_files, _ = spikee_extract_cli(run_spikee, workspace_dir, result_files=results_files, category="no-guardrail")

        assert len(extract_files) == 1
        extracted = read_jsonl_file(str(extract_files[0]))
        assert len(extracted) == 0

    def test_extract_error(self, run_spikee, workspace_dir):
        results_files, _ = self._run_test(run_spikee, workspace_dir, "always_error")
        extract_files, _ = spikee_extract_cli(run_spikee, workspace_dir, result_files=results_files, category="error")

        assert len(extract_files) == 1
        extracted = read_jsonl_file(str(extract_files[0]))
        assert len(extracted) > 0
        assert all(
            e.get("error") not in [None, "No response received"]
            for e in extracted
        )

    def test_extract_output_file_created(self, run_spikee, workspace_dir):
        results_files, _ = self._run_test(run_spikee, workspace_dir, "always_success")
        extract_files, _ = spikee_extract_cli(run_spikee, workspace_dir, result_files=results_files, category="success")

        assert len(extract_files) == 1
        filename = extract_files[0].name
        assert "extract" in filename
        assert "success" in filename

    def test_extract_traceability_long_id(self, run_spikee, workspace_dir):
        results_files, _ = self._run_test(run_spikee, workspace_dir, "always_success")
        source = results_files[0].stem  # filename without .jsonl
        extract_files, _ = spikee_extract_cli(run_spikee, workspace_dir, result_files=results_files, category="success")

        extracted = read_jsonl_file(str(extract_files[0]))
        assert all(e["long_id"].endswith(f"_extracted_{source}") for e in extracted)

    def test_extract_original_id_preserved(self, run_spikee, workspace_dir):
        results_files, _ = self._run_test(run_spikee, workspace_dir, "always_success")
        extract_files, _ = spikee_extract_cli(run_spikee, workspace_dir, result_files=results_files, category="success")

        extracted = read_jsonl_file(str(extract_files[0]))
        assert all("original_id" in e for e in extracted)

    def test_extract_custom_field_match(self, run_spikee, workspace_dir):
        results_files, entries = self._run_test(run_spikee, workspace_dir, "always_success")
        # always_success returns "EN-CHECK IT-CHECK restricted-check" — canary is in response
        extract_files, _ = spikee_extract_cli(
            run_spikee, workspace_dir,
            result_files=results_files,
            category="custom",
            custom_search=["response:EN-CHECK"],
        )

        extracted = read_jsonl_file(str(extract_files[0]))
        assert len(extracted) == len(entries)

    def test_extract_custom_inverted_no_match(self, run_spikee, workspace_dir):
        results_files, _ = self._run_test(run_spikee, workspace_dir, "always_success")
        extract_files, _ = spikee_extract_cli(
            run_spikee, workspace_dir,
            result_files=results_files,
            category="custom",
            custom_search=["response:!EN-CHECK"],
        )

        extracted = read_jsonl_file(str(extract_files[0]))
        assert len(extracted) == 0

    def test_extract_invalid_category_fails(self, run_spikee, workspace_dir):
        results_files, _ = self._run_test(run_spikee, workspace_dir, "always_success")
        with pytest.raises(subprocess.CalledProcessError):
            spikee_extract_cli(run_spikee, workspace_dir, result_files=results_files, category="invalid_xyz")

    def test_extract_custom_multi_both_match(self, run_spikee, workspace_dir):
        # Two conditions both satisfied by the canary response
        results_files, entries = self._run_test(run_spikee, workspace_dir, "always_success")
        extract_files, _ = spikee_extract_cli(
            run_spikee, workspace_dir,
            result_files=results_files,
            category="custom",
            custom_search=["response:EN-CHECK", "response:IT-CHECK"],
        )

        extracted = read_jsonl_file(str(extract_files[0]))
        assert len(extracted) == len(entries)

    def test_extract_custom_multi_second_fails(self, run_spikee, workspace_dir):
        # Second condition kills all matches — term not in any response
        results_files, _ = self._run_test(run_spikee, workspace_dir, "always_success")
        extract_files, _ = spikee_extract_cli(
            run_spikee, workspace_dir,
            result_files=results_files,
            category="custom",
            custom_search=["response:EN-CHECK", "response:ABSENT_TERM_XYZ"],
        )

        extracted = read_jsonl_file(str(extract_files[0]))
        assert len(extracted) == 0

    def test_extract_custom_multi_field_and_success(self, run_spikee, workspace_dir):
        # Cross-field: response term + success field (coerced to "True" by str())
        results_files, entries = self._run_test(run_spikee, workspace_dir, "always_success")
        extract_files, _ = spikee_extract_cli(
            run_spikee, workspace_dir,
            result_files=results_files,
            category="custom",
            custom_search=["response:EN-CHECK", "success:True"],
        )

        extracted = read_jsonl_file(str(extract_files[0]))
        assert len(extracted) == len(entries)

    def test_extract_custom_multi_inverted_plus_match(self, run_spikee, workspace_dir):
        # Normal condition passes AND inverted condition passes (absent term)
        results_files, entries = self._run_test(run_spikee, workspace_dir, "always_success")
        extract_files, _ = spikee_extract_cli(
            run_spikee, workspace_dir,
            result_files=results_files,
            category="custom",
            custom_search=["response:EN-CHECK", "response:!ABSENT_TERM_XYZ"],
        )

        extracted = read_jsonl_file(str(extract_files[0]))
        assert len(extracted) == len(entries)

    def test_extract_custom_multi_inverted_kills_all(self, run_spikee, workspace_dir):
        # Contradictory: normal passes but inverted of the same term fails
        results_files, _ = self._run_test(run_spikee, workspace_dir, "always_success")
        extract_files, _ = spikee_extract_cli(
            run_spikee, workspace_dir,
            result_files=results_files,
            category="custom",
            custom_search=["response:EN-CHECK", "response:!EN-CHECK"],
        )

        extracted = read_jsonl_file(str(extract_files[0]))
        assert len(extracted) == 0

    def test_extract_from_result_folder(self, run_spikee, workspace_dir):
        # Generate two separate datasets and run tests to populate the results folder
        dataset1 = spikee_generate_cli(run_spikee, workspace_dir)
        _, _ = spikee_test_cli(
            run_spikee, workspace_dir, target="always_success", datasets=[dataset1],
            additional_args=["--no-auto-resume"],
        )
        entries1 = read_jsonl_file(dataset1)

        dataset2 = spikee_generate_cli(run_spikee, workspace_dir)
        _, _ = spikee_test_cli(
            run_spikee, workspace_dir, target="always_success", datasets=[dataset2],
            additional_args=["--no-auto-resume"],
        )
        entries2 = read_jsonl_file(dataset2)

        results_folder = workspace_dir / "results"
        extract_files, _ = spikee_extract_cli(
            run_spikee, workspace_dir,
            result_files=[results_folder],
            category="success",
        )

        assert len(extract_files) == 1
        extracted = read_jsonl_file(str(extract_files[0]))
        assert len(extracted) == len(entries1) + len(entries2)

