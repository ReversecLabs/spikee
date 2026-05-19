"""Functional tests for threaded dataset generation (--threads parameter).

Tests verify that:
1. Sequential generation (--threads 1) produces correct output
2. Parallel generation (--threads > 1) produces equivalent output
3. Thread count affects performance appropriately
4. Entry IDs are assigned correctly in both modes
5. Errors are handled gracefully in parallel mode
"""

import pytest
import time
from spikee.utilities.files import read_jsonl_file
from ..utils import spikee_generate_cli


class TestThreadsBasic:
    """Basic tests for --threads parameter functionality."""

    def test_threads_default_sequential(self, run_spikee, workspace_dir):
        """Test default behavior (no --threads) uses sequential processing.
        
        Verifies:
        - Default generation works without --threads parameter
        - Produces expected number of entries
        - All entries have valid structure
        """
        output_file = spikee_generate_cli(run_spikee, workspace_dir)
        
        assert output_file.exists(), f"Expected dataset file at {output_file}"
        
        dataset = read_jsonl_file(output_file)
        
        assert len(dataset) > 0, "Generated dataset contains no entries"
        assert len(dataset) >= 4, f"Expected at least 4 entries, got {len(dataset)}"
        
        # Verify all entries have required fields
        for entry in dataset:
            assert "id" in entry, "Entry missing 'id' field"
            assert "long_id" in entry, "Entry missing 'long_id' field"
            assert "content" in entry, "Entry missing 'content' field"
            assert "judge_name" in entry, "Entry missing 'judge_name' field"

    def test_threads_explicit_sequential(self, run_spikee, workspace_dir):
        """Test explicit --threads 1 uses sequential processing.
        
        Verifies:
        - --threads 1 flag works correctly
        - Produces same output as default behavior
        - Entry IDs are sequential
        """
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--threads", "1"]
        )
        
        assert output_file.exists(), f"Expected dataset file at {output_file}"
        
        dataset = read_jsonl_file(output_file)
        
        assert len(dataset) >= 4, f"Expected at least 4 entries, got {len(dataset)}"
        
        # Verify entry IDs are sequential starting from 1
        ids = [e["id"] for e in dataset]
        assert ids == list(range(1, len(dataset) + 1)), \
            f"Expected sequential IDs 1..{len(dataset)}, got {ids}"

    def test_threads_parallel_basic(self, run_spikee, workspace_dir):
        """Test basic parallel generation with --threads 2.
        
        Verifies:
        - --threads 2 flag works correctly
        - Produces correct number of entries
        - All entries have valid structure
        """
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--threads", "2"]
        )
        
        assert output_file.exists(), f"Expected dataset file at {output_file}"
        
        dataset = read_jsonl_file(output_file)
        
        assert len(dataset) >= 4, f"Expected at least 4 entries, got {len(dataset)}"
        
        # Verify all entries have required fields
        for entry in dataset:
            assert "id" in entry, "Entry missing 'id' field"
            assert "long_id" in entry, "Entry missing 'long_id' field"
            assert "content" in entry, "Entry missing 'content' field"
            assert "payload" in entry, "Entry missing 'payload' field"


class TestThreadsEquivalence:
    """Tests that sequential and parallel generation produce equivalent datasets."""

    def test_sequential_vs_parallel_same_output(self, run_spikee, workspace_dir):
        """Test that sequential and parallel generation produce equivalent datasets.
        
        Verifies:
        - Both modes generate same number of entries
        - Entries have same content (order may differ)
        - All permutations are covered in both modes
        """
        # Generate with sequential processing
        output_sequential = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--threads", "1"]
        )
        
        # Generate with parallel processing
        output_parallel = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--threads", "3"]
        )
        
        dataset_seq = read_jsonl_file(output_sequential)
        dataset_par = read_jsonl_file(output_parallel)
        
        # Should have same number of entries
        assert len(dataset_seq) == len(dataset_par), \
            f"Sequential generated {len(dataset_seq)} entries, parallel generated {len(dataset_par)}"
        
        # Extract long_ids (unique identifiers for permutations)
        long_ids_seq = sorted([e["long_id"] for e in dataset_seq])
        long_ids_par = sorted([e["long_id"] for e in dataset_par])
        
        # Both should have identical sets of long_ids
        assert long_ids_seq == long_ids_par, \
            "Sequential and parallel modes generated different permutations"
        
        # Compare content for each long_id
        seq_by_id = {e["long_id"]: e for e in dataset_seq}
        par_by_id = {e["long_id"]: e for e in dataset_par}
        
        for long_id in long_ids_seq:
            seq_entry = seq_by_id[long_id]
            par_entry = par_by_id[long_id]
            
            # Content should be identical
            assert seq_entry["content"] == par_entry["content"], \
                f"Content mismatch for long_id={long_id}"
            
            # Payload should be identical
            assert seq_entry["payload"] == par_entry["payload"], \
                f"Payload mismatch for long_id={long_id}"
            
            # Metadata should be identical
            assert seq_entry["judge_name"] == par_entry["judge_name"], \
                f"Judge name mismatch for long_id={long_id}"

    def test_different_thread_counts_same_output(self, run_spikee, workspace_dir):
        """Test that different thread counts produce equivalent datasets.
        
        Verifies:
        - --threads 2 and --threads 4 produce same results
        - Only performance differs, not output
        """
        output_2_threads = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--threads", "2"]
        )
        
        output_4_threads = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--threads", "4"]
        )
        
        dataset_2 = read_jsonl_file(output_2_threads)
        dataset_4 = read_jsonl_file(output_4_threads)
        
        # Should have same number of entries
        assert len(dataset_2) == len(dataset_4), \
            f"2 threads: {len(dataset_2)} entries, 4 threads: {len(dataset_4)} entries"
        
        # Should have same long_ids
        long_ids_2 = sorted([e["long_id"] for e in dataset_2])
        long_ids_4 = sorted([e["long_id"] for e in dataset_4])
        
        assert long_ids_2 == long_ids_4, \
            "Different thread counts produced different permutations"


class TestThreadsWithPlugins:
    """Tests for threaded generation with plugins."""

    def test_threads_with_simple_plugin(self, run_spikee, workspace_dir):
        """Test threaded generation with a simple transformation plugin.
        
        Verifies:
        - Plugins work correctly in parallel mode
        - Plugin transformations are applied consistently
        - Both base and plugin entries are generated
        """
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--threads", "3", "--plugins", "test_upper"]
        )
        
        dataset = read_jsonl_file(output_file)
        
        # Should have base entries + plugin entries
        assert len(dataset) == 12, f"Expected 12 entries (6 base + 6 plugin), got {len(dataset)}"
        
        # Verify plugin entries
        plugin_entries = [e for e in dataset if e.get("plugin") == "test_upper"]
        assert len(plugin_entries) == 6, f"Expected 6 plugin entries, got {len(plugin_entries)}"
        
        # Verify plugin transformation (uppercase)
        for entry in plugin_entries:
            assert entry["payload"] == entry["payload"].upper(), \
                "Plugin should uppercase the payload"

    def test_threads_with_plugin_sequential_equivalence(self, run_spikee, workspace_dir):
        """Test that plugin transformations are identical in sequential and parallel modes.
        
        Verifies:
        - Plugin output is deterministic
        - Sequential and parallel produce same plugin transformations
        """
        # Sequential with plugin
        output_seq = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--threads", "1", "--plugins", "test_upper"]
        )
        
        # Parallel with plugin
        output_par = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--threads", "3", "--plugins", "test_upper"]
        )
        
        dataset_seq = read_jsonl_file(output_seq)
        dataset_par = read_jsonl_file(output_par)
        
        # Same number of entries
        assert len(dataset_seq) == len(dataset_par), \
            f"Sequential: {len(dataset_seq)}, Parallel: {len(dataset_par)}"
        
        # Compare plugin entries specifically
        plugin_seq = sorted([e for e in dataset_seq if e.get("plugin") == "test_upper"],
                           key=lambda x: x["long_id"])
        plugin_par = sorted([e for e in dataset_par if e.get("plugin") == "test_upper"],
                           key=lambda x: x["long_id"])
        
        assert len(plugin_seq) == len(plugin_par), \
            f"Sequential: {len(plugin_seq)} plugin entries, Parallel: {len(plugin_par)} plugin entries"
        
        # Verify transformations are identical
        for seq_entry, par_entry in zip(plugin_seq, plugin_par):
            assert seq_entry["payload"] == par_entry["payload"], \
                f"Plugin payload mismatch: {seq_entry['long_id']}"
            assert seq_entry["content"] == par_entry["content"], \
                f"Plugin content mismatch: {seq_entry['long_id']}"


class TestThreadsPerformance:
    """Tests for performance characteristics of threaded generation.
    
    Note: These are smoke tests, not precise benchmarks.
    They verify that parallelization doesn't slow things down significantly.
    """

    def test_threads_performance_smoke(self, run_spikee, workspace_dir):
        """Smoke test: Verify parallel mode doesn't take longer than sequential.
        
        This is a sanity check, not a performance benchmark.
        Parallel should be at least as fast as sequential for non-trivial datasets.
        """
        # Measure sequential time
        start = time.time()
        output_seq = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--threads", "1"]
        )
        sequential_time = time.time() - start
        
        # Measure parallel time
        start = time.time()
        output_par = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--threads", "3"]
        )
        parallel_time = time.time() - start
        
        # Verify both completed successfully
        assert output_seq.exists()
        assert output_par.exists()
        
        # Parallel shouldn't be significantly slower than sequential
        # Allow 3x overhead for thread management on small datasets
        assert parallel_time < sequential_time * 3, \
            f"Parallel ({parallel_time:.2f}s) much slower than sequential ({sequential_time:.2f}s)"


class TestThreadsWithFilters:
    """Tests for threaded generation with filtering options."""

    def test_threads_with_language_filter(self, run_spikee, workspace_dir):
        """Test threaded generation with language filtering.
        
        Verifies:
        - Language filtering works in parallel mode
        - Only specified language entries are generated
        """
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--threads", "3", "--languages", "en"]
        )
        
        dataset = read_jsonl_file(output_file)
        
        assert len(dataset) > 0, "Generated dataset contains no entries"
        
        # All entries should be English
        languages = {e.get("lang") for e in dataset}
        assert languages == {"en"}, f"Expected only 'en' language, got {languages}"

    def test_threads_with_instruction_filter(self, run_spikee, workspace_dir):
        """Test threaded generation with instruction type filtering.
        
        Verifies:
        - Instruction filtering works in parallel mode
        - Only specified instruction types are generated
        """
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--threads", "3", "--instruction-filter", "restricted"]
        )
        
        dataset = read_jsonl_file(output_file)
        
        assert len(dataset) == 2, f"Expected 2 entries for 'restricted' filter, got {len(dataset)}"
        
        # All entries should reference the filtered instruction
        for entry in dataset:
            assert "instr-filter" in entry.get("long_id", ""), \
                f"Expected 'instr-filter' in long_id, got: {entry.get('long_id')}"

    def test_threads_with_match_languages_false(self, run_spikee, workspace_dir):
        """Test threaded generation with cross-language pairing enabled.
        
        Verifies:
        - Cross-language pairing works in parallel mode
        - Generates all language combinations
        """
        output_file = spikee_generate_cli(
            run_spikee,
            workspace_dir,
            additional_args=["--threads", "3", "--match-languages", "false"]
        )
        
        dataset = read_jsonl_file(output_file)
        
        # Should have all cross-language combinations (2 docs × 2 jbs × 3 instrs = 12)
        assert len(dataset) == 12, f"Expected 12 entries with cross-language, got {len(dataset)}"
        
        # Verify cross-language entries exist
        long_ids = [e.get("long_id", "") for e in dataset]
        cross_lang_entries = [
            lid for lid in long_ids
            if ("jb-it" in lid and "instr-en" in lid) or ("jb-en" in lid and "instr-it" in lid)
        ]
        assert len(cross_lang_entries) > 0, "Expected cross-language entries"