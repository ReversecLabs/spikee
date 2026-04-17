"""
Functional tests for content validation against function signatures.

Tests the validation functions:
- validate_content_signature(): Validate content against function parameter type hints
- validate_content_annotation(): Validate content against type annotations
"""
import inspect
from typing import Union, Optional
import pytest

from spikee.utilities.hinting import (
    Content,
    Audio,
    Image,
    validate_content_signature,
    validate_content_annotation,
)


# Test functions with various type signatures
def function_str_only(llm_input: str) -> bool:
    """Function that only accepts str."""
    return True


def function_audio_only(llm_input: Audio) -> bool:
    """Function that only accepts Audio."""
    return True


def function_image_only(llm_input: Image) -> bool:
    """Function that only accepts Image."""
    return True


def function_content_union(llm_input: Content) -> bool:
    """Function that accepts Content (Union[str, Audio, Image])."""
    return True


def function_str_or_audio(llm_input: Union[str, Audio]) -> bool:
    """Function that accepts str or Audio."""
    return True


def function_no_type_hint(llm_input):
    """Legacy function with no type hints (backward compatibility)."""
    return True


def function_optional_str(llm_input: Optional[str]) -> bool:
    """Function with Optional[str] parameter."""
    return True


def function_multiple_params(llm_input: str, llm_output: str) -> bool:
    """Function with multiple parameters."""
    return True


def function_wrong_param_name(input_text: str) -> bool:
    """Function with different parameter name."""
    return True


class TestValidateContentSignature:
    """Test validate_content_signature() for parameter validation."""

    def test_validate_str_against_str_function(self):
        """str content should validate against str parameter."""
        assert validate_content_signature("Hello", function_str_only, "llm_input") is True

    def test_validate_audio_against_audio_function(self):
        """Audio content should validate against Audio parameter."""
        audio = Audio("audiodata")
        assert validate_content_signature(audio, function_audio_only, "llm_input") is True

    def test_validate_image_against_image_function(self):
        """Image content should validate against Image parameter."""
        image = Image("imagedata")
        assert validate_content_signature(image, function_image_only, "llm_input") is True

    def test_validate_str_against_content_union(self):
        """str should validate against Content union."""
        assert validate_content_signature("Hello", function_content_union, "llm_input") is True

    def test_validate_audio_against_content_union(self):
        """Audio should validate against Content union."""
        audio = Audio("audiodata")
        assert validate_content_signature(audio, function_content_union, "llm_input") is True

    def test_validate_image_against_content_union(self):
        """Image should validate against Content union."""
        image = Image("imagedata")
        assert validate_content_signature(image, function_content_union, "llm_input") is True

    def test_validate_audio_against_str_fails(self):
        """Audio should fail validation against str-only parameter."""
        audio = Audio("audiodata")
        assert validate_content_signature(audio, function_str_only, "llm_input") is False

    def test_validate_image_against_str_fails(self):
        """Image should fail validation against str-only parameter."""
        image = Image("imagedata")
        assert validate_content_signature(image, function_str_only, "llm_input") is False

    def test_validate_str_against_audio_fails(self):
        """str should fail validation against Audio-only parameter."""
        assert validate_content_signature("Hello", function_audio_only, "llm_input") is False

    def test_validate_partial_union(self):
        """Should validate against partial Union types."""
        # str should pass for Union[str, Audio]
        assert validate_content_signature("Hello", function_str_or_audio, "llm_input") is True

        # Audio should pass for Union[str, Audio]
        audio = Audio("audiodata")
        assert validate_content_signature(audio, function_str_or_audio, "llm_input") is True

        # Image should fail for Union[str, Audio]
        image = Image("imagedata")
        assert validate_content_signature(image, function_str_or_audio, "llm_input") is False

    def test_validate_no_type_hint_defaults_to_str(self):
        """Functions without type hints should default to str validation."""
        # str should pass
        assert validate_content_signature("Hello", function_no_type_hint, "llm_input") is True

        # Audio/Image should fail (defaults to str)
        assert validate_content_signature(Audio("data"), function_no_type_hint, "llm_input") is False
        assert validate_content_signature(Image("data"), function_no_type_hint, "llm_input") is False

    def test_validate_optional_str(self):
        """Should handle Optional[str] annotations."""
        # str should validate
        assert validate_content_signature("Hello", function_optional_str, "llm_input") is True

        # None is special case - handled by Optional
        # Audio/Image should fail (Optional[str] = Union[str, None])
        assert validate_content_signature(Audio("data"), function_optional_str, "llm_input") is False

    def test_validate_wrong_parameter_raises_error(self):
        """Non-existent parameter should raise ValueError."""
        with pytest.raises(ValueError, match="Parameter 'nonexistent' not found"):
            validate_content_signature("Hello", function_str_only, "nonexistent")

    def test_validate_multiple_params_checks_correct_one(self):
        """Should validate against the correct parameter."""
        # llm_input parameter should accept str
        assert validate_content_signature("Hello", function_multiple_params, "llm_input") is True

        # llm_output parameter should also accept str
        assert validate_content_signature("World", function_multiple_params, "llm_output") is True

        # Audio should fail for str-only parameters
        assert validate_content_signature(Audio("data"), function_multiple_params, "llm_input") is False


class TestValidateContentAnnotation:
    """Test validate_content_annotation() for direct annotation validation."""

    def test_validate_str_annotation(self):
        """str content should validate against str annotation."""
        assert validate_content_annotation("Hello", str) is True

    def test_validate_audio_annotation(self):
        """Audio content should validate against Audio annotation."""
        audio = Audio("audiodata")
        assert validate_content_annotation(audio, Audio) is True

    def test_validate_image_annotation(self):
        """Image content should validate against Image annotation."""
        image = Image("imagedata")
        assert validate_content_annotation(image, Image) is True

    def test_validate_union_annotation(self):
        """Should handle Union annotations."""
        # Content = Union[str, Audio, Image]
        assert validate_content_annotation("Hello", Content) is True
        assert validate_content_annotation(Audio("data"), Content) is True
        assert validate_content_annotation(Image("data"), Content) is True

    def test_validate_partial_union_annotation(self):
        """Should handle partial Union types."""
        str_or_audio = Union[str, Audio]

        assert validate_content_annotation("Hello", str_or_audio) is True
        assert validate_content_annotation(Audio("data"), str_or_audio) is True
        assert validate_content_annotation(Image("data"), str_or_audio) is False

    def test_validate_empty_annotation_defaults_to_str(self):
        """Empty annotation (inspect.Parameter.empty) should default to str."""
        assert validate_content_annotation("Hello", inspect.Parameter.empty) is True

        # Audio/Image should fail against default str
        assert validate_content_annotation(Audio("data"), inspect.Parameter.empty) is False
        assert validate_content_annotation(Image("data"), inspect.Parameter.empty) is False

    def test_validate_invalid_annotation_returns_false(self):
        """Invalid/unsupported annotation should return False (permissive)."""
        # Non-type annotation should return False
        assert validate_content_annotation("Hello", "not_a_type") is False
        assert validate_content_annotation("Hello", 12345) is False

    def test_validate_optional_annotation(self):
        """Should handle Optional annotations."""
        opt_str = Optional[str]

        # str should validate
        assert validate_content_annotation("Hello", opt_str) is True

        # Audio/Image should fail (Optional[str] doesn't include them)
        assert validate_content_annotation(Audio("data"), opt_str) is False


class TestBackwardCompatibility:
    """Test backward compatibility with legacy functions."""

    def test_legacy_judge_no_type_hints(self):
        """Legacy judges without type hints default to str validation."""
        def legacy_judge(llm_input, llm_output, judge_args):
            return True

        # Only str should pass (defaults to str)
        assert validate_content_signature("text", legacy_judge, "llm_input") is True

        # Audio/Image require explicit type hints
        assert validate_content_signature(Audio("data"), legacy_judge, "llm_input") is False
        assert validate_content_signature(Image("data"), legacy_judge, "llm_input") is False

    def test_mixed_typed_and_untyped_params(self):
        """Functions with mix of typed and untyped parameters."""
        def mixed_function(llm_input: str, llm_output, judge_args):
            return True

        # Typed parameter should validate strictly
        assert validate_content_signature("text", mixed_function, "llm_input") is True
        assert validate_content_signature(Audio("data"), mixed_function, "llm_input") is False

        # Untyped parameter defaults to str
        assert validate_content_signature("text", mixed_function, "llm_output") is True
        assert validate_content_signature(Audio("data"), mixed_function, "llm_output") is False

    def test_gradually_typed_migration(self):
        """Support gradual type hint migration."""
        # Start: No type hints (defaults to str)
        def v1_function(llm_input):
            return True

        # Intermediate: Explicit str type hints
        def v2_function(llm_input: str):
            return True

        # Final: Full type hints with Content union
        def v3_function(llm_input: Content):
            return True

        # v1 defaults to str, same as v2
        assert validate_content_signature("text", v1_function, "llm_input") is True
        assert validate_content_signature(Audio("data"), v1_function, "llm_input") is False

        # v2 explicitly str only
        assert validate_content_signature("text", v2_function, "llm_input") is True
        assert validate_content_signature(Audio("data"), v2_function, "llm_input") is False

        # v3 should accept all Content types
        assert validate_content_signature("text", v3_function, "llm_input") is True
        assert validate_content_signature(Audio("data"), v3_function, "llm_input") is True
        assert validate_content_signature(Image("data"), v3_function, "llm_input") is True


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_none_content(self):
        """None should not validate as Content."""
        assert validate_content_annotation(None, str) is False
        assert validate_content_annotation(None, Content) is False

    def test_empty_string_validates(self):
        """Empty string should still validate as str."""
        assert validate_content_signature("", function_str_only, "llm_input") is True

    def test_numeric_content_fails(self):
        """Numeric types should fail validation."""
        assert validate_content_annotation(12345, str) is False
        assert validate_content_annotation(12345, Content) is False

    def test_list_content_fails(self):
        """List should fail validation."""
        assert validate_content_annotation(["item"], str) is False
        assert validate_content_annotation(["item"], Content) is False

    def test_dict_content_fails(self):
        """Dict should fail validation."""
        assert validate_content_annotation({"key": "value"}, str) is False
        assert validate_content_annotation({"key": "value"}, Content) is False
