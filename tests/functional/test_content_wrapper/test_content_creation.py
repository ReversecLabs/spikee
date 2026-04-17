"""
Functional tests for content creation, extraction, and type detection.

Tests the core content wrapper functions:
- content_factory(): Create Content objects from raw data
- get_content(): Extract raw content from Content wrappers
- get_content_type(): Determine content type
"""
import base64
import pytest

from spikee.utilities.hinting import (
    Audio,
    Image,
    content_factory,
    get_content,
    get_content_type,
)


class TestContentFactory:
    """Test content_factory() for creating Content objects."""

    def test_factory_text_creates_string(self):
        """Text type should return raw string."""
        result = content_factory("Hello world", content_type="text")
        assert isinstance(result, str)
        assert result == "Hello world"

    def test_factory_audio_creates_audio_wrapper(self):
        """Audio type should return Audio wrapper."""
        result = content_factory("base64audiodata", content_type="audio")
        assert isinstance(result, Audio)
        assert result.content == "base64audiodata"

    def test_factory_image_creates_image_wrapper(self):
        """Image type should return Image wrapper."""
        result = content_factory("base64imagedata", content_type="image")
        assert isinstance(result, Image)
        assert result.content == "base64imagedata"

    def test_factory_case_insensitive(self):
        """Factory should accept uppercase type strings."""
        text_result = content_factory("test", content_type="TEXT")
        audio_result = content_factory("data", content_type="AUDIO")
        image_result = content_factory("data", content_type="IMAGE")

        assert isinstance(text_result, str)
        assert isinstance(audio_result, Audio)
        assert isinstance(image_result, Image)

    def test_factory_invalid_type_raises_error(self):
        """Invalid content type should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported content type: video"):
            content_factory("data", content_type="video")

    def test_factory_default_type_is_text(self):
        """Default content type should be text."""
        result = content_factory("Hello world")
        assert isinstance(result, str)
        assert result == "Hello world"

    def test_factory_preserves_complex_content(self):
        """Factory should preserve complex base64 strings."""
        # Sample base64-encoded data
        sample_b64 = base64.b64encode(b"Complex binary data here").decode()

        audio = content_factory(sample_b64, content_type="audio")
        image = content_factory(sample_b64, content_type="image")

        assert audio.content == sample_b64
        assert image.content == sample_b64

    def test_factory_empty_string(self):
        """Factory should handle empty strings."""
        text = content_factory("", content_type="text")
        audio = content_factory("", content_type="audio")
        image = content_factory("", content_type="image")

        assert text == ""
        assert audio.content == ""
        assert image.content == ""


class TestGetContent:
    """Test get_content() for extracting raw content."""

    def test_extract_from_string(self):
        """Should extract string content directly."""
        result = get_content("Hello world")
        assert result == "Hello world"

    def test_extract_from_audio(self):
        """Should extract content from Audio wrapper."""
        audio = Audio("base64audiodata")
        result = get_content(audio)
        assert result == "base64audiodata"

    def test_extract_from_image(self):
        """Should extract content from Image wrapper."""
        image = Image("base64imagedata")
        result = get_content(image)
        assert result == "base64imagedata"

    def test_extract_preserves_content(self):
        """Extracted content should be identical to original."""
        original = "Complex content with special chars: !@#$%^&*()"

        text = get_content(content_factory(original, "text"))
        audio = get_content(content_factory(original, "audio"))
        image = get_content(content_factory(original, "image"))

        assert text == original
        assert audio == original
        assert image == original

    def test_extract_unsupported_type_raises_error(self):
        """Unsupported type should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported content type"):
            get_content(12345)  # Integer is not a Content type

    def test_extract_empty_content(self):
        """Should handle empty content."""
        assert get_content("") == ""
        assert get_content(Audio("")) == ""
        assert get_content(Image("")) == ""

    def test_extract_multiline_content(self):
        """Should preserve multiline content."""
        multiline = "Line 1\nLine 2\nLine 3"

        assert get_content(multiline) == multiline
        assert get_content(Audio(multiline)) == multiline
        assert get_content(Image(multiline)) == multiline


class TestGetContentType:
    """Test get_content_type() for type detection."""

    def test_detect_string_type(self):
        """Should detect string content as 'text'."""
        result = get_content_type("Hello world")
        assert result == "text"

    def test_detect_audio_type(self):
        """Should detect Audio wrapper as 'audio'."""
        audio = Audio("base64audiodata")
        result = get_content_type(audio)
        assert result == "audio"

    def test_detect_image_type(self):
        """Should detect Image wrapper as 'image'."""
        image = Image("base64imagedata")
        result = get_content_type(image)
        assert result == "image"

    def test_detect_factory_created_types(self):
        """Should correctly detect types from factory-created content."""
        text = content_factory("data", "text")
        audio = content_factory("data", "audio")
        image = content_factory("data", "image")

        assert get_content_type(text) == "text"
        assert get_content_type(audio) == "audio"
        assert get_content_type(image) == "image"

    def test_unsupported_type_raises_error(self):
        """Unsupported type should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported content type"):
            get_content_type(12345)

    def test_detect_empty_content_types(self):
        """Should detect types even with empty content."""
        assert get_content_type("") == "text"
        assert get_content_type(Audio("")) == "audio"
        assert get_content_type(Image("")) == "image"


class TestImageBase64Inline:
    """Test Image.base64_inline() method."""

    def test_base64_inline_format(self):
        """Should return proper data URI format."""
        image = Image("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
        result = image.base64_inline()

        assert result.startswith("data:image/png;base64,")
        assert "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" in result

    def test_base64_inline_preserves_content(self):
        """Inline format should preserve base64 content."""
        original_b64 = "VGVzdCBkYXRh"
        image = Image(original_b64)
        result = image.base64_inline()

        assert original_b64 in result
        assert result == f"data:image/png;base64,{original_b64}"


class TestContentRoundTrip:
    """Test complete create → extract → type-detect cycle."""

    @pytest.mark.parametrize("content_type,expected_type", [
        ("text", "text"),
        ("audio", "audio"),
        ("image", "image"),
    ])
    def test_roundtrip_preserves_data(self, content_type, expected_type):
        """Content should survive create → extract → type-detect cycle."""
        original = "Sample content data"

        # Create
        created = content_factory(original, content_type)

        # Type detect
        detected_type = get_content_type(created)
        assert detected_type == expected_type

        # Extract
        extracted = get_content(created)
        assert extracted == original

    def test_roundtrip_with_complex_base64(self):
        """Should handle complex base64 data."""
        # Real base64-encoded PNG (1x1 red pixel)
        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

        image = content_factory(png_b64, "image")
        assert get_content_type(image) == "image"
        assert get_content(image) == png_b64


class TestProcessTargetContent:
    """Test process_target_content() helper."""

    def test_unwraps_str_content(self):
        """Plain str response returned as-is."""
        from spikee.utilities.hinting import process_target_content
        assert process_target_content("hello") == "hello"

    def test_unwraps_audio_content(self):
        """Audio response returns raw content string."""
        from spikee.utilities.hinting import process_target_content
        assert process_target_content(Audio("audio_data")) == "audio_data"

    def test_unwraps_image_content(self):
        """Image response returns raw content string."""
        from spikee.utilities.hinting import process_target_content
        assert process_target_content(Image("image_data")) == "image_data"

    def test_unwraps_tuple_str(self):
        """(str, meta) tuple unpacks and returns str."""
        from spikee.utilities.hinting import process_target_content
        assert process_target_content(("hello", {"tokens": 5})) == "hello"

    def test_unwraps_tuple_audio(self):
        """(Audio, meta) tuple unpacks and returns raw content."""
        from spikee.utilities.hinting import process_target_content
        assert process_target_content((Audio("audio_data"), None)) == "audio_data"

    def test_unwraps_tuple_image(self):
        """(Image, meta) tuple unpacks and returns raw content."""
        from spikee.utilities.hinting import process_target_content
        assert process_target_content((Image("image_data"), None)) == "image_data"

    def test_bool_response_raises(self):
        """bool response raises ValueError (guardrail mode not handled here)."""
        from spikee.utilities.hinting import process_target_content
        with pytest.raises((ValueError, TypeError)):
            process_target_content(True)

    def test_wrong_tuple_length_raises(self):
        """Tuple with != 2 elements raises ValueError."""
        from spikee.utilities.hinting import process_target_content
        with pytest.raises(ValueError):
            process_target_content(("a", "b", "c"))
