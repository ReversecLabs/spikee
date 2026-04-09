"""Unit tests for spikee.generator functions using the generate_dataset entry point."""

import pytest

from spikee.utilities.files import read_jsonl_file
from ..utils import spikee_generate_cli


class TestSourceArguments:
    def test_seed_folder(self, run_spikee, workspace_dir):
        """Test basic seed folder loading with default arguments.

        Verifies:
        - Basic dataset generation with default settings
        - Standalone inputs are excluded by default
        - System messages are excluded by default
        """
        output_file = spikee_generate_cli(run_spikee, workspace_dir)

        assert output_file.exists(), f"Expected dataset file at {output_file}, but it does not exist."

        # Load and verify dataset
        dataset = read_jsonl_file(output_file)

        assert len(dataset) > 0, "Generated dataset contains no entries"
        # seeds-functional-basic: 2 docs × 2 jailbreaks × 2 matching instructions (en-en, it-it) = 4 entries
        assert len(dataset) >= 4, f"Expected at least 4 entries, got {len(dataset)}"

        # Verify standalone inputs are excluded by default
        standalone_entries = [e for e in dataset if e.get("document_id") is None]
        assert len(standalone_entries) == 0, f"Expected 0 standalone entries by default, got {len(standalone_entries)}"

        # Verify system messages are excluded by default
        system_messages = {e.get("system_message") for e in dataset}
        assert system_messages == {None}, f"Expected all system_message to be None by default, got {system_messages}"

    def test_seed_folder_invalid(self, run_spikee, workspace_dir):
        """Test that generate fails gracefully with non-existent seed folder."""
        with pytest.raises(Exception):
            spikee_generate_cli(
                run_spikee,
                workspace_dir,
                seed_folder="datasets/invalid-folder-xyz",
            )

    def test_include_standalone_inputs_flag(self, run_spikee, workspace_dir):
        """Test --include-standalone-inputs adds standalone entries."""
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--include-standalone-inputs"],
        )

        assert output_file.exists(), f"Expected dataset file at {output_file}"

        # Load and verify dataset
        dataset = read_jsonl_file(output_file)

        # Should have base entries (4) + 2 standalone entries
        assert len(dataset) >= 6, f"Expected at least 6 entries with standalone, got {len(dataset)}"

        # Verify standalone entries exist (document_id should be None)
        standalone_entries = [e for e in dataset if e.get("document_id") is None]
        assert len(standalone_entries) == 2, f"Expected 2 standalone entries, got {len(standalone_entries)}"

    def test_include_system_message_flag(self, run_spikee, workspace_dir):
        """Test --include-system-message adds system messages to entries."""
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--include-system-message"],
        )

        assert output_file.exists(), f"Expected dataset file at {output_file}"

        # Load and verify dataset
        dataset = read_jsonl_file(output_file)

        assert len(dataset) > 0, "Generated dataset contains no entries"

        # Verify system_message field is present in entries
        system_messages = {e.get("system_message") for e in dataset}
        # Should have at least one non-None system message
        non_none_messages = {m for m in system_messages if m is not None}
        assert len(non_none_messages) > 0, "Expected at least one system message in entries with --include-system-message"


class TestFormattingArguments:
    """Test cases for formatting arguments: --format, --positions, --injection-delimiters, --languages, --match-languages"""

    def test_format_full_prompt(self, run_spikee, workspace_dir):
        """Test --format full-prompt generates correct format."""
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--format", "full-prompt"],
        )

        assert output_file.exists(), f"Expected dataset file at {output_file}"

        dataset = read_jsonl_file(output_file)

        assert len(dataset) == 12, f"Expected 12 entries, got {len(dataset)}"

        # full-prompt produces summarization and qna task types only
        task_types = {e.get("task_type") for e in dataset}
        assert task_types == {"summarization", "qna"}, f"Expected task_types 'summarization' and 'qna', got {task_types}"

        # Summarization entries: text starts with "Summarize..." and have ideal_summary
        summarization_entries = [e for e in dataset if e.get("task_type") == "summarization"]
        for entry in summarization_entries:
            assert entry["text"].startswith("Summarize the following document:"), \
                f"Summarization text should start with 'Summarize the following document:', got: {entry['text'][:60]}"
            assert "ideal_summary" in entry, "Summarization entry missing 'ideal_summary' field"

        # QnA entries: text starts with "Given this document:" and have ideal_answer
        qna_entries = [e for e in dataset if e.get("task_type") == "qna"]
        for entry in qna_entries:
            assert entry["text"].startswith("Given this document:"), \
                f"QnA text should start with 'Given this document:', got: {entry['text'][:60]}"
            assert "ideal_answer" in entry, "QnA entry missing 'ideal_answer' field"

        # system_message should be None without --include-system-message
        assert all(e.get("system_message") is None for e in dataset), \
            "Expected all system_message to be None without --include-system-message"

    def test_format_full_prompt_with_system_message(self, run_spikee, workspace_dir):
        """Test --format full-prompt with --include-system-message populates system_message on all entries."""
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--format", "full-prompt", "--include-system-message"],
        )

        assert output_file.exists(), f"Expected dataset file at {output_file}"

        dataset = read_jsonl_file(output_file)

        assert len(dataset) == 12, f"Expected 12 entries, got {len(dataset)}"

        # All entries must have a non-None system_message
        assert all(e.get("system_message") is not None for e in dataset), \
            "Expected all entries to have a system_message with --include-system-message"

        # long_id should contain '-sys' suffix for all entries
        assert all("-sys" in e.get("long_id", "") for e in dataset), \
            "Expected all long_id values to contain '-sys' suffix"

    def test_position_start(self, run_spikee, workspace_dir):
        """Test --positions start injects payload before the document text."""
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--positions", "start"],
        )

        dataset = read_jsonl_file(output_file)

        assert len(dataset) == 6, f"Expected 6 entries, got {len(dataset)}"

        positions = {e.get("position") for e in dataset}
        assert positions == {"start"}, f"Expected position 'start', got {positions}"

        # At 'start', the payload appears before the document text in 'text'
        for entry in dataset:
            payload = entry["payload"]
            text = entry["text"]
            doc_snippet = "This is the base document" if "This is the base document" in text \
                          else "Documento to base"
            assert text.index(payload) < text.index(doc_snippet), \
                "Expected payload before document text for position 'start'"

    def test_position_end(self, run_spikee, workspace_dir):
        """Test --positions end injects payload after the document text (default)."""
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--positions", "end"],
        )

        dataset = read_jsonl_file(output_file)

        assert len(dataset) == 6, f"Expected 6 entries, got {len(dataset)}"

        positions = {e.get("position") for e in dataset}
        assert positions == {"end"}, f"Expected position 'end', got {positions}"

        # At 'end', the payload appears after the document text in 'text'
        for entry in dataset:
            payload = entry["payload"]
            text = entry["text"]
            doc_pos = text.find("This is the base document") if "This is the base document" in text \
                else text.find("Documento to base")
            assert text.index(payload) > doc_pos, \
                "Expected payload after document text for position 'end'"

    def test_position_middle(self, run_spikee, workspace_dir):
        """Test --positions middle injects payload in the middle of the document text."""
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--positions", "middle"],
        )

        dataset = read_jsonl_file(output_file)

        assert len(dataset) == 6, f"Expected 6 entries, got {len(dataset)}"

        positions = {e.get("position") for e in dataset}
        assert positions == {"middle"}, f"Expected position 'middle', got {positions}"

        # At 'middle', the payload appears between parts of the document
        for entry in dataset:
            payload = entry["payload"]
            text = entry["text"]
            assert payload in text, "Expected payload in text for position 'middle'"

    def test_placeholder_position(self, run_spikee, workspace_dir):
        """Test placeholder documents inject payload at <PLACEHOLDER> and set position to 'fixed'."""
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            seed_folder="datasets/seeds-functional-placeholder",
        )

        dataset = read_jsonl_file(output_file)

        assert len(dataset) > 0, "Generated dataset contains no entries"

        # All entries should have position 'fixed' (placeholder overrides positions arg)
        positions = {e.get("position") for e in dataset}
        assert positions == {"fixed"}, f"Expected all positions to be 'fixed', got {positions}"

        # The payload should be injected where <PLACEHOLDER> was in the source document
        # Source: "User start <PLACEHOLDER> user end"
        for entry in dataset:
            text = entry["text"]
            payload = entry["payload"]
            assert "User start" in text, "Expected 'User start' in text"
            assert "user end" in text, "Expected 'user end' in text"
            assert payload in text, "Expected payload in text"
            # <PLACEHOLDER> should be replaced, not literally present
            assert "<PLACEHOLDER>" not in text, "Expected <PLACEHOLDER> to be replaced in text"

    def test_injection_delimiters_custom(self, run_spikee, workspace_dir):
        """Test --injection-delimiters wraps the payload in the generated text.

        The delimiter pattern must contain 'INJECTION_PAYLOAD' as a token, which
        gets replaced with the actual jailbreak text. Everything else in the pattern
        becomes literal text surrounding the payload.
        """
        custom_delimiter = "<<<INJECTION_PAYLOAD>>>"
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--injection-delimiters", custom_delimiter],
        )

        assert output_file.exists(), f"Expected dataset file at {output_file}"

        dataset = read_jsonl_file(output_file)

        assert len(dataset) > 0, "Generated dataset contains no entries"

        # Verify the pattern is stored as-is in the injection_delimiters field
        stored_delimiters = {e.get("injection_delimiters") for e in dataset}
        assert stored_delimiters == {custom_delimiter}, \
            f"Expected injection_delimiters '{custom_delimiter}', got {stored_delimiters}"

        # Verify the delimiter actually wraps the payload in the generated text:
        # '<<<' should appear immediately before the payload, '>>>' immediately after
        for entry in dataset:
            text = entry["text"]
            payload = entry["payload"]
            assert f"<<<{payload}>>>" in text, \
                f"Expected payload wrapped in '<<<...>>>' in text, but got: {text[:120]}"

    def test_languages_filter_english(self, run_spikee, workspace_dir):
        """Test --languages en filters to English entries only."""
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--languages", "en"],
        )

        assert output_file.exists(), f"Expected dataset file at {output_file}"

        # Load and verify dataset
        dataset = read_jsonl_file(output_file)

        assert len(dataset) > 0, "Generated dataset contains no entries"

        # Verify all entries have English language
        languages = {e.get("lang") for e in dataset}
        assert languages == {"en"}, f"Expected only 'en' language, got {languages}"

    def test_languages_filter_italian(self, run_spikee, workspace_dir):
        """Test --languages it filters to Italian entries only."""
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--languages", "it"],
        )

        assert output_file.exists(), f"Expected dataset file at {output_file}"

        # Load and verify dataset
        dataset = read_jsonl_file(output_file)

        assert len(dataset) > 0, "Generated dataset contains no entries"

        # Verify all entries have Italian language
        languages = {e.get("lang") for e in dataset}
        assert languages == {"it"}, f"Expected only 'it' language, got {languages}"

    def test_match_languages_false(self, run_spikee, workspace_dir):
        """Test --match-languages false generates cross-language jailbreak+instruction pairs.
        """
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--match-languages", "false"],
        )

        assert output_file.exists(), f"Expected dataset file at {output_file}"

        dataset = read_jsonl_file(output_file)

        assert len(dataset) == 12, f"Expected 12 entries (all cross-language combos), got {len(dataset)}"

        # Cross-language pairs must exist: e.g. Italian jailbreak paired with English instruction
        long_ids = [e.get("long_id", "") for e in dataset]
        cross_lang_entries = [
            lid for lid in long_ids
            if ("jb-it" in lid and ("instr-en" in lid or "instr-filter" in lid))
            or ("jb-en" in lid and "instr-it" in lid)
        ]
        assert len(cross_lang_entries) > 0, \
            "Expected cross-language entries with --match-languages false, but none found"


class TestFixes:
    """Test cases for prefixes and suffixes"""

    def test_adv_prefixes(self, run_spikee, workspace_dir):
        """Test that adversarial prefixes from adv_prefixes.jsonl are applied correctly."""
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--include-fixes", "adv_prefixes"]
        )

        assert output_file.exists(), f"Expected dataset file at {output_file}"

        dataset = read_jsonl_file(output_file)

        assert len(dataset) > 0, "Generated dataset contains no entries"

        # The generator produces entries with and without the prefix (None baseline + prefix).
        # Filter to entries that actually have a prefix applied.
        prefixed_entries = [e for e in dataset if e.get("prefix_id") is not None]
        assert len(prefixed_entries) > 0, "Expected at least one entry with a prefix applied"

        for entry in prefixed_entries:
            payload = entry["payload"]
            assert payload.startswith("#-PREFIX-#"), \
                f"Expected payload to start with '#-PREFIX-#', got: {payload[:80]}"

    def test_adv_suffixes(self, run_spikee, workspace_dir):
        """Test that adversarial suffixes from adv_suffixes.jsonl are applied correctly."""
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--include-fixes", "adv_suffixes"]
        )

        assert output_file.exists(), f"Expected dataset file at {output_file}"

        dataset = read_jsonl_file(output_file)

        assert len(dataset) > 0, "Generated dataset contains no entries"

        # The generator produces entries with and without the suffix (None baseline + suffix).
        # Filter to entries that actually have a suffix applied.
        suffixed_entries = [e for e in dataset if e.get("suffix_id") is not None]
        assert len(suffixed_entries) > 0, "Expected at least one entry with a suffix applied"

        for entry in suffixed_entries:
            payload = entry["payload"]
            assert payload.endswith("#-SUFFIX-#"), \
                f"Expected payload to end with '#-SUFFIX-#', got: {payload[-80:]}"

    def test_custom_prefix(self, run_spikee, workspace_dir):
        """Test that a custom prefix specified via --custom-prefix is applied correctly."""
        custom_prefix = "<<<CUSTOM_PREFIX>>>"
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--include-fixes", f"prefix={custom_prefix}"]
        )

        assert output_file.exists(), f"Expected dataset file at {output_file}"

        dataset = read_jsonl_file(output_file)

        assert len(dataset) > 0, "Generated dataset contains no entries"

        # The generator produces entries with and without the prefix (None baseline + prefix).
        # Filter to entries that actually have the custom prefix applied.
        prefixed_entries = [e for e in dataset if e.get("prefix_id") is not None]
        assert len(prefixed_entries) > 0, "Expected at least one entry with a custom prefix applied"

        for entry in prefixed_entries:
            payload = entry["payload"]
            assert payload.startswith(custom_prefix), \
                f"Expected payload to start with '{custom_prefix}', got: {payload[:80]}"

    def test_custom_suffix(self, run_spikee, workspace_dir):
        """Test that a custom suffix specified via --custom-suffix is applied correctly."""
        custom_suffix = "<<<CUSTOM_SUFFIX>>>"
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--include-fixes", f"suffix={custom_suffix}"]
        )

        assert output_file.exists(), f"Expected dataset file at {output_file}"

        dataset = read_jsonl_file(output_file)

        assert len(dataset) > 0, "Generated dataset contains no entries"

        # The generator produces entries with and without the suffix (None baseline + suffix).
        # Filter to entries that actually have the custom suffix applied.
        suffixed_entries = [e for e in dataset if e.get("suffix_id") is not None]
        assert len(suffixed_entries) > 0, "Expected at least one entry with a custom suffix applied"

        for entry in suffixed_entries:
            payload = entry["payload"]
            assert payload.endswith(custom_suffix), \
                f"Expected payload to end with '{custom_suffix}', got: {payload[-80:]}"

    def test_adv_fix_combination(self, run_spikee, workspace_dir):
        """Test that combining multiple fixes (e.g. adv_prefixes and adv_suffixes) applies all of them correctly."""
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--include-fixes", "adv_prefixes,adv_suffixes"]
        )

        assert output_file.exists(), f"Expected dataset file at {output_file}"

        dataset = read_jsonl_file(output_file)

        assert len(dataset) > 0, "Generated dataset contains no entries"

        # The generator produces a cartesian product of prefix × suffix (including None baselines).
        # Filter to entries that have both a prefix and a suffix applied.
        fixed_entries = [e for e in dataset if e.get("prefix_id") is not None and e.get("suffix_id") is not None]
        assert len(fixed_entries) > 0, "Expected at least one entry with both prefix and suffix applied"

        for entry in fixed_entries:
            payload = entry["payload"]
            assert payload.startswith("#-PREFIX-#"), \
                f"Expected payload to start with '#-PREFIX-#', got: {payload[:80]}"
            assert payload.endswith("#-SUFFIX-#"), \
                f"Expected payload to end with '#-SUFFIX-#', got: {payload[-80:]}"


class TestFilteringArguments:
    """Test cases for filtering arguments: --instruction-filter, --jailbreak-filter"""

    def test_instruction_filter_single_type(self, run_spikee, workspace_dir):
        """Test --instruction-filter restricts to only the specified instruction type.
        """
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--instruction-filter", "restricted"],
        )

        dataset = read_jsonl_file(output_file)

        assert len(dataset) == 2, f"Expected 2 entries for 'restricted' instruction type, got {len(dataset)}"

        # Every long_id should reference instr-filter
        for entry in dataset:
            assert "instr-filter" in entry.get("long_id", ""), \
                f"Expected 'instr-filter' in long_id but got: {entry.get('long_id')}"

    def test_jailbreak_filter_single_type(self, run_spikee, workspace_dir):
        """Test --jailbreak-filter restricts to only the specified jailbreak type.
        """
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--jailbreak-filter", "test"],
        )

        dataset = read_jsonl_file(output_file)

        assert len(dataset) == 4, f"Expected 4 entries for 'test' jailbreak type, got {len(dataset)}"

        # Every long_id should reference jb-en (the 'test'-type jailbreak)
        for entry in dataset:
            assert "jb-en" in entry.get("long_id", ""), \
                f"Expected 'jb-en' in long_id but got: {entry.get('long_id')}"

    def test_instruction_and_jailbreak_filter_combined(self, run_spikee, workspace_dir):
        """Test combining --instruction-filter and --jailbreak-filter.
        """
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--instruction-filter", "exfil", "--jailbreak-filter", "dev"],
        )

        dataset = read_jsonl_file(output_file)

        assert len(dataset) == 2, f"Expected 2 entries with combined filters, got {len(dataset)}"

        for entry in dataset:
            long_id = entry.get("long_id", "")
            assert "jb-it" in long_id, f"Expected 'jb-it' in long_id but got: {long_id}"
            assert "instr-it" in long_id, f"Expected 'instr-it' in long_id but got: {long_id}"


class TestPlugins:
    """Test cases for plugin application arguments: --plugins, --plugin-options and --plugin-only"""

    def test_plugins_individual(self, run_spikee, workspace_dir):
        """Test --plugins with three individual plugins (test_upper, base64, 1337).

        Each plugin is applied independently to each base entry. With 6 base combos
        and 3 plugins, the dataset has 6 base entries + 6×3 plugin entries = 24 total.
        """
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--plugins", "test_upper", "base64", "1337"],
        )

        dataset = read_jsonl_file(output_file)

        assert len(dataset) == 24, f"Expected 24 entries (6 base + 6×3 plugin), got {len(dataset)}"

        # Check each plugin is represented
        plugin_names = {e.get("plugin") for e in dataset}
        assert None in plugin_names, "Expected base entries (plugin=None)"
        assert "test_upper" in plugin_names, "Expected test_upper plugin entries"
        assert "base64" in plugin_names, "Expected base64 plugin entries"
        assert "1337" in plugin_names, "Expected 1337 plugin entries"

        # test_upper entries should have uppercased payload
        upper_entries = [e for e in dataset if e.get("plugin") == "test_upper"]
        assert len(upper_entries) == 6, f"Expected 6 test_upper entries, got {len(upper_entries)}"
        for entry in upper_entries:
            assert entry["payload"] == entry["payload"].upper(), \
                f"Expected uppercase payload for test_upper plugin, got: {entry['payload'][:60]}"

        # Plugin name appears in long_id via plugin_suffix
        upper_long_ids = [e["long_id"] for e in upper_entries]
        assert all("_test_upper-1" in lid for lid in upper_long_ids), \
            "Expected '_test_upper-1' in all test_upper long_ids"

        base64_long_ids = [e["long_id"] for e in dataset if e.get("plugin") == "base64"]
        assert all("_base64-1" in lid for lid in base64_long_ids), \
            "Expected '_base64-1' in all base64 long_ids"

        leet_long_ids = [e["long_id"] for e in dataset if e.get("plugin") == "1337"]
        assert all("_1337-1" in lid for lid in leet_long_ids), \
            "Expected '_1337-1' in all 1337 long_ids"

    def test_piped_plugins(self, run_spikee, workspace_dir):
        """Test --plugins with a piped chain test_upper|base64|1337.

        Piped plugins are applied sequentially as a single combined plugin.
        With 6 base combos and 1 piped plugin, the dataset has 6 base + 6 piped = 12 entries.
        The piped plugin name in the dataset uses '~' as the separator.
        """
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--plugins", "test_upper|base64|1337"],
        )

        dataset = read_jsonl_file(output_file)

        assert len(dataset) == 12, f"Expected 12 entries (6 base + 6 piped), got {len(dataset)}"

        # Piped plugin entries should use '~' as separator in plugin name
        piped_entries = [e for e in dataset if e.get("plugin") is not None]
        assert len(piped_entries) == 6, f"Expected 6 piped plugin entries, got {len(piped_entries)}"
        piped_plugin_names = {e.get("plugin") for e in piped_entries}
        assert piped_plugin_names == {"test_upper~base64~1337"}, \
            f"Expected piped plugin name 'test_upper~base64~1337', got {piped_plugin_names}"

        # long_id should embed the piped plugin name
        for entry in piped_entries:
            assert "test_upper~base64~1337" in entry["long_id"], \
                f"Expected 'test_upper~base64~1337' in long_id, got: {entry['long_id']}"

    def test_plugin_options_repeat(self, run_spikee, workspace_dir):
        """Test --plugin-options with test_repeat and n_variants=3.

        test_repeat with n_variants=3 produces 3 variants per base combo.
        With 6 base combos: 6 base + 3×6 plugin = 24 total entries.
        Variant long_ids are suffixed _test_repeat-1, _test_repeat-2, _test_repeat-3.
        """
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=[
                "--plugins", "test_repeat",
                "--plugin-options", "test_repeat:n_variants=3",
            ],
        )

        dataset = read_jsonl_file(output_file)

        assert len(dataset) == 24, f"Expected 24 entries (6 base + 3×6 plugin), got {len(dataset)}"

        repeat_entries = [e for e in dataset if e.get("plugin") == "test_repeat"]
        assert len(repeat_entries) == 18, f"Expected 18 test_repeat entries, got {len(repeat_entries)}"

        # All three variant indices must be present
        repeat_long_ids = [e["long_id"] for e in repeat_entries]
        assert any("_test_repeat-1" in lid for lid in repeat_long_ids), "Missing _test_repeat-1 variant"
        assert any("_test_repeat-2" in lid for lid in repeat_long_ids), "Missing _test_repeat-2 variant"
        assert any("_test_repeat-3" in lid for lid in repeat_long_ids), "Missing _test_repeat-3 variant"

        # Second variant payload should contain the default suffix '-repeat'
        variant2_entries = [e for e in repeat_entries if "_test_repeat-2" in e["long_id"]]
        for entry in variant2_entries:
            assert entry["payload"].endswith("-repeat"), \
                f"Expected payload ending in '-repeat' for variant 2, got: {entry['payload']}"

    def test_inference_plugin_invalid_model(self, run_spikee, workspace_dir):
        """Test test_inference plugin with an invalid model name fails gracefully."""
        with pytest.raises(Exception):
            spikee_generate_cli(
                run_spikee,
                workspace_dir,
                additional_args=[
                    "--plugins", "test_inference",
                    "--plugin-options", "test_inference:model=openai/nonexistent-model-xyz",
                    "--plugin-only",
                    "--languages", "en",
                ],
            )

    def test_legacy_plugins(self, run_spikee, workspace_dir):
        """Test that legacy function-based plugins (test_upper_legacy, test_repeat_legacy) still work.

        Legacy plugins produce the same output as their OOP equivalents.
        """
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--plugins", "test_upper_legacy", "test_repeat_legacy"],
        )

        dataset = read_jsonl_file(output_file)

        # 6 base + 6 test_upper_legacy (1 var) + 12 test_repeat_legacy (2 var default) = 24
        assert len(dataset) == 24, f"Expected 24 entries, got {len(dataset)}"

        # Legacy test_upper: payload must be uppercase
        upper_legacy_entries = [e for e in dataset if e.get("plugin") == "test_upper_legacy"]
        assert len(upper_legacy_entries) == 6, \
            f"Expected 6 test_upper_legacy entries, got {len(upper_legacy_entries)}"
        for entry in upper_legacy_entries:
            assert entry["payload"] == entry["payload"].upper(), \
                f"Expected uppercase payload for test_upper_legacy, got: {entry['payload'][:60]}"

        # Legacy test_repeat: 2 variants per combo (default n_variants=2)
        repeat_legacy_entries = [e for e in dataset if e.get("plugin") == "test_repeat_legacy"]
        assert len(repeat_legacy_entries) == 12, \
            f"Expected 12 test_repeat_legacy entries, got {len(repeat_legacy_entries)}"
        variant2_entries = [e for e in repeat_legacy_entries if "_test_repeat_legacy-2" in e["long_id"]]
        assert len(variant2_entries) == 6, f"Expected 6 variant-2 entries, got {len(variant2_entries)}"
        for entry in variant2_entries:
            assert entry["payload"].endswith("-repeat"), \
                f"Expected payload ending in '-repeat' for legacy repeat variant 2, got: {entry['payload']}"

    def test_plugin_only(self, run_spikee, workspace_dir):
        """Test --plugin-only suppresses base entries and outputs only plugin-transformed entries.

        With --plugins test_upper (1 variant per combo) and 6 base combos,
        --plugin-only produces exactly 6 entries and no base (un-transformed) entries.
        """
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--plugins", "test_upper", "--plugin-only"],
        )

        dataset = read_jsonl_file(output_file)

        assert len(dataset) == 6, f"Expected 6 plugin-only entries, got {len(dataset)}"

        # All entries must be plugin entries — no base (plugin=None) entries
        assert all(e.get("plugin") is not None for e in dataset), \
            "Expected no base entries with --plugin-only"
        assert all(e.get("plugin") == "test_upper" for e in dataset), \
            "Expected all entries to have plugin 'test_upper'"

        # All payloads should be uppercased
        for entry in dataset:
            assert entry["payload"] == entry["payload"].upper(), \
                f"Expected uppercase payload in plugin-only mode, got: {entry['payload'][:60]}"
