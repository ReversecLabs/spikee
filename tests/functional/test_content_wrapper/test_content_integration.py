"""
Integration tests for Content wrapper across the entire pipeline.

Tests Content flow through:
- Target process_input() with Content types
- Plugin transform() with Content
- Judge validation with Content
- Generator Entry class integration
- Tester end-to-end flow
"""
import json
import os
from contextlib import contextmanager

import pytest

from spikee.utilities.hinting import (
    Audio,
    Image,
    get_content,
    get_content_type,
    validate_content_signature,
)
from spikee.utilities.files import read_jsonl_file
from spikee.utilities.modules import load_module_from_path


@contextmanager
def working_directory(path):
    """Context manager to temporarily change working directory."""
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


class TestTargetIntegration:
    """Test Content integration with target modules."""

    def test_audio_target_accepts_audio_input(self, workspace_dir):
        """Audio target should accept Audio input only."""
        with working_directory(workspace_dir):
            target = load_module_from_path("mock_audio_target", "targets")

        audio_input = Audio("test_audio_data")
        result = target.process_input(audio_input)

        assert isinstance(result, Audio)
        assert "AUDIO_ECHO" in get_content(result)
        assert "test_audio_data" in get_content(result)

    def test_image_target_accepts_image_input(self, workspace_dir):
        """Image target should accept Image input only."""
        with working_directory(workspace_dir):
            target = load_module_from_path("mock_image_target", "targets")

        image_input = Image("base64imagedata")
        result = target.process_input(image_input)

        assert isinstance(result, Image)
        assert "IMAGE_ECHO" in get_content(result)
        assert "base64imagedata" in get_content(result)

    def test_multimodal_target_preserves_input_type(self, workspace_dir):
        """Multimodal target should return same type as input."""
        with working_directory(workspace_dir):
            target = load_module_from_path("mock_multimodal_target", "targets")

        # Test with text
        text_result = target.process_input("text_data")
        assert isinstance(text_result, str)
        assert "MULTIMODAL_ECHO[text]" in text_result

        # Test with Audio
        audio_result = target.process_input(Audio("audio_data"))
        assert isinstance(audio_result, Audio)
        assert "MULTIMODAL_ECHO[audio]" in get_content(audio_result)

        # Test with Image
        image_result = target.process_input(Image("image_data"))
        assert isinstance(image_result, Image)
        assert "MULTIMODAL_ECHO[image]" in get_content(image_result)

    def test_target_signature_validation(self, workspace_dir):
        """Targets should validate correctly against their signatures."""
        with working_directory(workspace_dir):
            audio_target = load_module_from_path("mock_audio_target", "targets")
            image_target = load_module_from_path("mock_image_target", "targets")
            multimodal_target = load_module_from_path("mock_multimodal_target", "targets")

        # Audio target only accepts Audio
        assert validate_content_signature(Audio("data"), audio_target.process_input, "input_text") is True
        assert validate_content_signature("text", audio_target.process_input, "input_text") is False
        assert validate_content_signature(Image("data"), audio_target.process_input, "input_text") is False

        # Image target only accepts Image
        assert validate_content_signature(Image("data"), image_target.process_input, "input_text") is True
        assert validate_content_signature("text", image_target.process_input, "input_text") is False
        assert validate_content_signature(Audio("data"), image_target.process_input, "input_text") is False

        # Multimodal target accepts any Content type
        assert validate_content_signature("text", multimodal_target.process_input, "input_text") is True
        assert validate_content_signature(Audio("data"), multimodal_target.process_input, "input_text") is True
        assert validate_content_signature(Image("data"), multimodal_target.process_input, "input_text") is True

    def test_plugin_transform(self, workspace_dir):
        """Plugin should transform text content correctly."""
        with working_directory(workspace_dir):
            plugin = load_module_from_path("uppercase_content", "plugins")

        assert plugin.transform("hello world") == "HELLO WORLD"


class TestJudgeIntegration:
    """Test Content integration with judge modules."""

    def test_content_type_judge_accepts_content(self, workspace_dir):
        """Content type judge accepts any Content type and checks for marker."""
        with working_directory(workspace_dir):
            judge = load_module_from_path("content_type_judge", "judges")

        # Works with str input/output
        assert judge.judge("input", "AUDIO_ECHO response", "AUDIO_ECHO") is True
        assert judge.judge("input", "IMAGE_ECHO response", "IMAGE_ECHO") is True

        # Works with Audio input/output
        assert judge.judge(Audio("in"), Audio("AUDIO_ECHO output"), "AUDIO_ECHO") is True

        # Signature accepts all Content types
        assert validate_content_signature("text", judge.judge, "llm_input") is True
        assert validate_content_signature(Audio("data"), judge.judge, "llm_input") is True
        assert validate_content_signature(Image("data"), judge.judge, "llm_input") is True

    def test_audio_only_judge_strict_typing(self, workspace_dir):
        """Audio-only judge requires Audio types and rejects all others."""
        with working_directory(workspace_dir):
            judge = load_module_from_path("audio_only_judge", "judges")

        assert judge.judge(Audio("input"), Audio("expected_output"), "expected") is True

        # Signature accepts Audio only
        assert validate_content_signature(Audio("test"), judge.judge, "llm_input") is True
        assert validate_content_signature(Audio("test"), judge.judge, "llm_output") is True
        assert validate_content_signature("text", judge.judge, "llm_input") is False
        assert validate_content_signature("text", judge.judge, "llm_output") is False
        assert validate_content_signature(Image("data"), judge.judge, "llm_input") is False


class TestGeneratorIntegration:
    """Test Content integration with generator Entry.to_entry() serialization."""

    def _make_entry(self, content, payload):
        from spikee.generator import Entry, EntryType
        return Entry(
            entry_type=EntryType.ATTACK,
            entry_id="e1",
            base_id="b1",
            jailbreak_id="jb1",
            instruction_id="inst1",
            prefix_id=None,
            suffix_id=None,
            content=content,
            entry_text=None,
            system_message=None,
            payload=payload,
            lang="en",
            plugin_suffix="",
            plugin_name=None,
            judge_name="canary",
            judge_args="FLAG",
            position="start",
            jailbreak_type=None,
            instruction_type=None,
            injection_pattern=None,
            spotlighting_data_markers=None,
        )

    def test_to_entry_text_content_type(self):
        """to_entry() should serialize text content with content_type='text'."""
        output = self._make_entry("some text", "payload").to_entry()
        assert output["content"] == "some text"
        assert output["content_type"] == "text"

    def test_to_entry_audio_content_type(self):
        """to_entry() should serialize Audio content with content_type='audio'."""
        output = self._make_entry(Audio("base64audio"), Audio("jailbreak")).to_entry()
        assert output["content"] == "base64audio"
        assert output["content_type"] == "audio"

    def test_to_entry_image_content_type(self):
        """to_entry() should serialize Image content with content_type='image'."""
        output = self._make_entry(Image("base64image"), Image("jailbreak")).to_entry()
        assert output["content"] == "base64image"
        assert output["content_type"] == "image"


class TestTesterIntegration:
    """Test Content integration with tester end-to-end flow."""

    def test_tester_with_audio_target(self, run_spikee, workspace_dir):
        """Tester should handle Audio target end-to-end."""
        from ..utils import spikee_test_cli

        # Create test dataset with text content
        dataset_path = workspace_dir / "datasets" / "test_audio_dataset.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = [
            {
                "id": "test_1",
                "long_id": "test_1",
                "content": "test input",
                "content_type": "audio",
                "payload": "test payload",
                "judge_name": "content_type_judge",
                "judge_args": "AUDIO_ECHO",
            }
        ]

        with open(dataset_path, "w") as f:
            for entry in dataset:
                f.write(json.dumps(entry) + "\n")

        # Run test with audio target and content judge
        results_path, result = spikee_test_cli(
            run_spikee,
            workspace_dir,
            target="mock_audio_target",
            datasets=[dataset_path],
            additional_args=["--judge", "content_type_judge"]
        )

        # Verify results
        results = read_jsonl_file(results_path[0])
        assert results, "No results recorded"

        # Should succeed - audio target returns Audio with AUDIO_ECHO marker
        assert all(entry["success"] for entry in results), \
            f"Expected all to succeed, got: {[(e['long_id'], e['success']) for e in results]}"

    def test_tester_with_multimodal_target(self, run_spikee, workspace_dir):
        """Tester should handle multimodal target end-to-end."""
        from ..utils import spikee_test_cli

        # Create test dataset
        dataset_path = workspace_dir / "datasets" / "test_multimodal_dataset.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = [
            {
                "id": "test_text",
                "long_id": "test_text",
                "content": "text content",
                "payload": "text payload",
                "judge_name": "content_type_judge",
                "judge_args": "MULTIMODAL_ECHO",
            }
        ]

        with open(dataset_path, "w") as f:
            for entry in dataset:
                f.write(json.dumps(entry) + "\n")

        # Run test with multimodal target
        results_path, result = spikee_test_cli(
            run_spikee,
            workspace_dir,
            target="mock_multimodal_target",
            datasets=[dataset_path],
            additional_args=["--judge", "content_type_judge"]
        )

        # Verify results
        results = read_jsonl_file(results_path[0])
        assert results, "No results recorded"
        assert all(entry["success"] for entry in results)

    def test_content_flow_through_pipeline(self, run_spikee, workspace_dir):
        """Test complete content flow: dataset → target → judge → results."""
        from ..utils import spikee_test_cli

        # Create dataset with specific content
        dataset_path = workspace_dir / "datasets" / "test_pipeline_dataset.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = [
            {
                "id": "pipeline_test",
                "long_id": "pipeline_test",
                "content": "pipeline input",
                "content_type": "image",
                "payload": "pipeline payload",
                "judge_name": "content_type_judge",
                "judge_args": "IMAGE_ECHO",
            }
        ]

        with open(dataset_path, "w") as f:
            for entry in dataset:
                f.write(json.dumps(entry) + "\n")

        # Run with image target
        results_path, result = spikee_test_cli(
            run_spikee,
            workspace_dir,
            target="mock_image_target",
            datasets=[dataset_path],
            additional_args=["--judge", "content_type_judge"]
        )

        # Verify results contain expected data
        results = read_jsonl_file(results_path[0])
        assert len(results) == 1

        entry = results[0]
        assert entry["success"] is True
        assert "IMAGE_ECHO" in entry["response"]
        assert "pipeline input" in entry["response"]


class TestMultiTurnIntegration:
    """Test Content with multi-turn conversations."""

    def test_multiturn_with_content_types(self):
        """Multi-turn should preserve Content types across messages."""
        from spikee.templates.standardised_conversation import StandardisedConversation

        conv = StandardisedConversation()

        # Add messages with different content types
        msg1 = conv.add_message(parent_id=0, data="First turn text", attempt=True)
        msg2 = conv.add_message(parent_id=msg1, data=Audio("Second turn audio"), attempt=True)
        msg3 = conv.add_message(parent_id=msg2, data=Image("Third turn image"), attempt=True)

        # Should have 3 messages plus root
        assert len(conv.conversation) == 4  # root + 3 messages

        # Verify data is preserved
        assert conv.get_message_data(msg1) == "First turn text"
        assert isinstance(conv.get_message_data(msg2), Audio)
        assert isinstance(conv.get_message_data(msg3), Image)


class TestEdgeCaseIntegration:
    """Test edge cases in Content integration."""

    def test_empty_content_through_pipeline(self, workspace_dir):
        """Empty content should flow through pipeline."""
        with working_directory(workspace_dir):
            target = load_module_from_path("mock_multimodal_target", "targets")

        # Empty string
        result = target.process_input("")
        assert get_content(result) == "MULTIMODAL_ECHO[text]:"

        # Empty Audio
        result = target.process_input(Audio(""))
        assert "MULTIMODAL_ECHO[audio]:" in get_content(result)

    def test_large_content_through_pipeline(self, workspace_dir):
        """Large content should flow through pipeline."""
        with working_directory(workspace_dir):
            target = load_module_from_path("mock_multimodal_target", "targets")

        # Large base64 string
        large_content = "A" * 10000
        result = target.process_input(Image(large_content))

        assert get_content_type(result) == "image"
        assert large_content in get_content(result)

    def test_special_characters_in_content(self, workspace_dir):
        """Special characters should be preserved."""
        with working_directory(workspace_dir):
            target = load_module_from_path("mock_multimodal_target", "targets")

        special_content = "!@#$%^&*(){}[]|\\:;\"'<>,.?/~`"
        result = target.process_input(Audio(special_content))

        assert special_content in get_content(result)


class TestCallJudgeContent:
    """Unit tests for call_judge() with Content-typed responses."""

    def test_call_judge_with_audio_response(self, workspace_dir):
        """call_judge() should pass Audio response to judge without stripping wrapper."""
        from spikee.judge import call_judge

        entry = {
            "judge_name": "content_type_judge",
            "judge_args": "AUDIO_ECHO",
            "judge_options": None,
            "content": Audio("test input"),
        }
        with working_directory(workspace_dir):
            result = call_judge(entry, Audio("AUDIO_ECHO[audio]:test input"))
        assert result is True

    def test_call_judge_with_image_response(self, workspace_dir):
        """call_judge() should pass Image response to judge correctly."""
        from spikee.judge import call_judge

        entry = {
            "judge_name": "content_type_judge",
            "judge_args": "IMAGE_ECHO",
            "judge_options": None,
            "content": Image("test input"),
        }
        with working_directory(workspace_dir):
            result = call_judge(entry, Image("IMAGE_ECHO[image]:test input"))
        assert result is True

    def test_call_judge_type_mismatch_raises(self, workspace_dir):
        """call_judge() should raise ValueError when content type doesn't match judge signature."""
        from spikee.judge import call_judge

        entry = {
            "judge_name": "audio_only_judge",
            "judge_args": "expected",
            "judge_options": None,
            "content": "plain text input",
        }
        with working_directory(workspace_dir):
            with pytest.raises(ValueError, match="do not match judge function signature"):
                call_judge(entry, "plain text response")

    def test_call_judge_bool_passthrough(self, workspace_dir):
        """call_judge() with bool output bypasses judge entirely."""
        from spikee.judge import call_judge

        entry = {"judge_name": "audio_only_judge", "judge_args": "x", "judge_options": None, "content": "x"}
        with working_directory(workspace_dir):
            assert call_judge(entry, True) is True
            assert call_judge(entry, False) is False

    def test_call_judge_empty_response_returns_false(self, workspace_dir):
        """call_judge() with empty string returns False without calling judge."""
        from spikee.judge import call_judge

        entry = {"judge_name": "content_type_judge", "judge_args": "x", "judge_options": None, "content": "x"}
        with working_directory(workspace_dir):
            assert call_judge(entry, "") is False
