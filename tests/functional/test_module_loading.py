"""
Test cases for module loading system edge cases (utilities/modules.py).

Focus on high-impact scenarios:
- Missing dependency error messages
- OOP vs legacy module precedence
- Malformed option strings
"""
import pytest
import os

from spikee.utilities.modules import (
    load_module_from_path,
    parse_options,
    get_default_option,
)


class TestModuleLoadingErrors:
    """Test error handling in module loading."""

    def test_missing_dependency_error_message(self, tmp_path):
        """Test that missing dependencies produce clear error messages."""
        # Create a module that imports a non-existent package
        module_dir = tmp_path / "targets"
        module_dir.mkdir()

        module_file = module_dir / "broken_import.py"
        module_file.write_text("""
from spikee.templates.target import Target
import nonexistent_package_xyz

class BrokenTarget(Target):
    def get_available_option_values(self):
        return [], False
    
    def process_input(self, input_text, system_message=None):
        return "response"
""")

        # Change to tmp directory to make module discoverable
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            with pytest.raises(ImportError) as exc_info:
                load_module_from_path("broken_import", "targets")

            error_msg = str(exc_info.value)
            # Should mention the missing dependency clearly
            assert "nonexistent_package_xyz" in error_msg or "dependency" in error_msg.lower()

        finally:
            os.chdir(original_cwd)

    def test_module_not_found_error_message(self):
        """Test that non-existent modules produce helpful error messages."""
        with pytest.raises(ImportError) as exc_info:
            load_module_from_path("definitely_does_not_exist_xyz", "targets")

        error_msg = str(exc_info.value)
        # Should suggest using 'spikee list' command
        assert "spikee list" in error_msg
        assert "definitely_does_not_exist_xyz" in error_msg

    def test_oop_vs_legacy_precedence(self, tmp_path):
        """Test that OOP class takes precedence over legacy function in same module."""
        module_dir = tmp_path / "targets"
        module_dir.mkdir()

        # Create module with both OOP class and legacy function
        module_file = module_dir / "hybrid_module.py"
        module_file.write_text("""
from spikee.templates.target import Target

# OOP implementation
class HybridTarget(Target):
    def get_available_option_values(self):
        return ["oop"], False
    
    def process_input(self, input_text, system_message=None):
        return "OOP_RESPONSE"

# Legacy function (should be ignored)
def process_input(input_text, system_message=None):
    return "LEGACY_RESPONSE"
""")

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            module = load_module_from_path("hybrid_module", "targets")

            # Should be OOP instance, not legacy module
            assert hasattr(module, "process_input")
            result = module.process_input("test input")

            # OOP implementation should be used
            assert result == "OOP_RESPONSE", "OOP class should take precedence over legacy function"

        finally:
            os.chdir(original_cwd)


class TestOptionParsing:
    """Test option string parsing edge cases."""

    def test_parse_options_double_equals(self):
        """Test parsing of malformed option string with double equals."""
        # Should handle gracefully - likely splits on first '='
        result = parse_options("key==value")
        # Either parses as {'key': '=value'} or skips malformed entry
        # Both behaviors are acceptable as long as no crash
        assert isinstance(result, dict)

    def test_parse_options_equals_only(self):
        """Test parsing of malformed option string with only equals."""
        result = parse_options("=value")
        # Should handle gracefully - either empty key or skip
        assert isinstance(result, dict)

    def test_parse_options_trailing_equals(self):
        """Test parsing of option string with trailing equals."""
        result = parse_options("key=")
        # Should parse as key with empty value
        assert isinstance(result, dict)
        if "key" in result:
            assert result["key"] == ""

    def test_parse_options_multiple_valid_and_invalid(self):
        """Test parsing of mixed valid and invalid options."""
        result = parse_options("valid=1,=invalid,another=2")
        assert isinstance(result, dict)
        # Valid options should be parsed correctly
        assert "valid" in result
        assert result["valid"] == "1"

    def test_parse_options_empty_string(self):
        """Test parsing of empty option string."""
        result = parse_options("")
        assert result == {}

    def test_parse_options_none(self):
        """Test parsing of None option string."""
        result = parse_options(None)
        assert result == {}


class TestDefaultOptions:
    """Test default option extraction."""

    def test_get_default_option_with_non_tuple(self, tmp_path):
        """Test that non-tuple returns from get_available_option_values are handled gracefully."""
        module_dir = tmp_path / "targets"
        module_dir.mkdir()

        module_file = module_dir / "bad_options.py"
        module_file.write_text("""
from spikee.templates.target import Target

class BadOptionsTarget(Target):
    def get_available_option_values(self):
        # Returns list instead of tuple - wrong type
        return ["option1", "option2"]
    
    def process_input(self, input_text, system_message=None):
        return "response"
""")

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            module = load_module_from_path("bad_options", "targets")
            default = get_default_option(module)

            # Should handle gracefully - either return None or extract first element
            # Current implementation checks isinstance(available, tuple) so returns None
            assert default is None or default == "option1"

        finally:
            os.chdir(original_cwd)
