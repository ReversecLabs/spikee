"""Test cases for builder utility functions from spikee.generator."""

import pytest
from spikee.generator import (
    insert_jailbreak,
)


class TestInsertJailbreak:
    """Test the insert_jailbreak function."""

    def test_insert_jailbreak_start_position(self):
        """Test inserting jailbreak at the start of document."""
        document = "This is the original document."
        jailbreak = "ATTACK_TEXT"
        pattern = "INJECTION_PAYLOAD"
        result = insert_jailbreak(document, jailbreak, "start", pattern, None)
        
        assert result.startswith("ATTACK_TEXT")
        assert result.endswith("This is the original document.")

    def test_insert_jailbreak_end_position(self):
        """Test inserting jailbreak at the end of document."""
        document = "This is the original document."
        jailbreak = "ATTACK_TEXT"
        pattern = "INJECTION_PAYLOAD"
        result = insert_jailbreak(document, jailbreak, "end", pattern, None)
        
        assert result.startswith("This is the original document.")
        assert result.endswith("ATTACK_TEXT")

    def test_insert_jailbreak_middle_position(self):
        """Test inserting jailbreak at the middle of document."""
        document = "This is the original document text content here."
        jailbreak = "ATTACK"
        pattern = "INJECTION_PAYLOAD"
        result = insert_jailbreak(document, jailbreak, "middle", pattern, None)
        
        # Should contain both original text and jailbreak
        assert "This is the original" in result
        assert "ATTACK" in result
        assert "document text content here." in result

    def test_insert_jailbreak_with_placeholder(self):
        """Test inserting jailbreak using placeholder replacement."""
        document = "This is text with <<MARKER>> placeholder."
        jailbreak = "INJECTED_CONTENT"
        pattern = "INJECTION_PAYLOAD"
        placeholder = "<<MARKER>>"
        result = insert_jailbreak(document, jailbreak, "start", pattern, placeholder)
        
        assert "<<MARKER>>" not in result
        assert "INJECTED_CONTENT" in result
        assert "This is text with" in result

    def test_insert_jailbreak_pattern_transformation(self):
        """Test that injection pattern is applied during insertion."""
        document = "Original document"
        jailbreak = "JAILBREAK"
        pattern = "[INJECTION_PAYLOAD]"  # Custom pattern with brackets
        result = insert_jailbreak(document, jailbreak, "start", pattern, None)
        
        # Pattern should transform jailbreak
        assert "[JAILBREAK]" in result

    def test_insert_jailbreak_missing_placeholder_raises_error(self):
        """Test that missing INJECTION_PAYLOAD in pattern raises error."""
        document = "Original"
        jailbreak = "ATTACK"
        pattern = "NO_PLACEHOLDER_HERE"
        
        with pytest.raises(ValueError, match="INJECTION_PAYLOAD"):
            insert_jailbreak(document, jailbreak, "start", pattern, None)

    def test_insert_jailbreak_invalid_position_raises_error(self):
        """Test invalid position raises error."""
        with pytest.raises(ValueError, match="Invalid position"):
            insert_jailbreak("doc", "jb", "invalid", "INJECTION_PAYLOAD", None)

