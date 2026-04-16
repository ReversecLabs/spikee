"""Test cases for builder utility functions from spikee.generator."""

import base64 as base64_lib
import os

import pytest

from spikee.generator import (
    parse_plugin_piping,
    parse_plugin_options,
    load_plugins,
    apply_plugin
)

class TestParsePluginPiping:
    """Test the parse_plugin_piping function."""

    def test_parse_plugin_piping_single_plugin(self):
        """Test parsing single plugin name."""
        result = parse_plugin_piping("plugin1")
        assert result == "plugin1"

    def test_parse_plugin_piping_multiple_plugins(self):
        """Test parsing multiple plugins with pipe separator."""
        result = parse_plugin_piping("plugin1|plugin2|plugin3")
        assert isinstance(result, list)
        assert result == ["plugin1", "plugin2", "plugin3"]

    def test_parse_plugin_piping_with_whitespace(self):
        """Test parsing handles whitespace around pipes."""
        result = parse_plugin_piping("plugin1 | plugin2 | plugin3")
        assert result == ["plugin1", "plugin2", "plugin3"]

    def test_parse_plugin_piping_none_returns_none(self):
        """Test that None returns None."""
        result = parse_plugin_piping(None)
        assert result is None

    def test_parse_plugin_piping_empty_string_returns_none(self):
        """Test that empty string returns None."""
        result = parse_plugin_piping("")
        assert result is None

class TestParsePluginOptions:
    """Test the parse_plugin_options function."""

    def test_parse_plugin_options_single_plugin(self):
        """Test parsing options for single plugin."""
        result = parse_plugin_options("plugin1:option_value")
        assert result == {"plugin1": "option_value"}

    def test_parse_plugin_options_multiple_plugins(self):
        """Test parsing options for multiple plugins."""
        result = parse_plugin_options("plugin1:opt1;plugin2:opt2;plugin3:opt3")
        assert result == {
            "plugin1": "opt1",
            "plugin2": "opt2",
            "plugin3": "opt3"
        }

    def test_parse_plugin_options_with_complex_values(self):
        """Test parsing options with complex values."""
        result = parse_plugin_options("plugin1:key=value,key2=value2;plugin2:mode=test")
        assert result == {
            "plugin1": "key=value,key2=value2",
            "plugin2": "mode=test"
        }

    def test_parse_plugin_options_none_returns_empty_dict(self):
        """Test that None returns empty dict."""
        result = parse_plugin_options(None)
        assert result == {}

    def test_parse_plugin_options_empty_string_returns_empty_dict(self):
        """Test that empty string returns empty dict."""
        result = parse_plugin_options("")
        assert result == {}

    def test_parse_plugin_options_missing_colon_ignored(self):
        """Test that entries without colon are ignored."""
        result = parse_plugin_options("plugin1:opt1;invalid_entry;plugin2:opt2")
        assert result == {
            "plugin1": "opt1",
            "plugin2": "opt2"
        }

class TestLoadPlugins:
    """Test the load_plugins function with real plugins."""

    def test_load_plugins_single_plugin_base64(self):
        """Test loading a single real plugin: base64."""
        result = load_plugins(["base64"])
        
        assert len(result) == 1
        assert result[0][0] == "base64"
        assert hasattr(result[0][1], "transform")

    def test_load_plugins_single_plugin_hex(self):
        """Test loading a single real plugin: hex."""
        result = load_plugins(["hex"])
        
        assert len(result) == 1
        assert result[0][0] == "hex"
        assert hasattr(result[0][1], "transform")

    def test_load_plugins_single_plugin_1337(self):
        """Test loading a single real plugin: 1337."""
        result = load_plugins(["1337"])
        
        assert len(result) == 1
        assert result[0][0] == "1337"
        assert hasattr(result[0][1], "transform")

    def test_load_plugins_multiple_plugins(self):
        """Test loading multiple real plugins."""
        result = load_plugins(["base64", "hex", "1337"])
        
        assert len(result) == 3
        assert result[0][0] == "base64"
        assert result[1][0] == "hex"
        assert result[2][0] == "1337"

    def test_load_plugins_piped_plugins(self):
        """Test loading plugins with piping syntax."""
        result = load_plugins(["base64|hex"])
        
        assert len(result) == 1
        assert result[0][0] == "base64~hex"
        assert len(result[0][1]) == 2
        assert result[0][1][0][0] == "base64"
        assert result[0][1][1][0] == "hex"

    def test_load_plugins_mixed_single_and_piped(self):
        """Test loading mix of single and piped plugins."""
        result = load_plugins(["1337", "base64|hex"])
        
        assert len(result) == 2
        assert result[0][0] == "1337"
        assert result[1][0] == "base64~hex"

    def test_load_plugins_empty_list(self):
        """Test loading empty plugin list."""
        result = load_plugins([])
        
        assert result == []

    def test_load_plugins_invalid_name_exits(self):
        """load_plugins calls sys.exit(1) when a plugin cannot be found."""
        with pytest.raises(SystemExit):
            load_plugins(["nonexistent_plugin_xyz_abc"])

class TestApplyPlugin:
    """Test apply_plugin with OOP plugins, legacy plugins, options, piping, and exclude patterns
    """

    def test_upper_basic(self, workspace_dir):
        """test_upper transforms text to uppercase, returns a single-element list."""
        os.chdir(workspace_dir)  # Ensure we're in the workspace for plugin loading

        plugins = load_plugins(["test_upper"])
        plugin_name, plugin_module = plugins[0]

        result = apply_plugin(plugin_name, plugin_module, "hello", None, None)

        assert isinstance(result, list)
        assert result == ["HELLO"]

    def test_1337_known_values(self, workspace_dir):
        """1337 plugin applies the fixed leet dictionary substitution."""
        os.chdir(workspace_dir)  # Ensure we're in the workspace for plugin loading

        plugins = load_plugins(["1337"])
        plugin_name, plugin_module = plugins[0]

        result = apply_plugin(plugin_name, plugin_module, "hello", None, None)

        assert isinstance(result, list)
        assert "h3ll0" in result

    def test_upper_legacy_matches_oop(self, workspace_dir):
        """test_upper_legacy (module-level function) produces the same output as the OOP version."""
        os.chdir(workspace_dir)  # Ensure we're in the workspace for plugin loading

        oop_plugins = load_plugins(["test_upper"])
        legacy_plugins = load_plugins(["test_upper_legacy"])

        text = "Hello World"
        oop_result = apply_plugin(*oop_plugins[0], text, None, None)
        legacy_result = apply_plugin(*legacy_plugins[0], text, None, None)

        assert oop_result == legacy_result == ["HELLO WORLD"]

    def test_repeat_legacy_default(self, workspace_dir):
        """test_repeat_legacy (module-level function) returns 2 variants by default."""
        os.chdir(workspace_dir)  # Ensure we're in the workspace for plugin loading

        plugins = load_plugins(["test_repeat_legacy"])
        plugin_name, plugin_module = plugins[0]

        result = apply_plugin(plugin_name, plugin_module, "payload", None, None)

        assert isinstance(result, list)
        assert result == ["payload", "payload-repeat"]

    def test_repeat_legacy_matches_oop(self, workspace_dir):
        """test_repeat_legacy produces the same output as test_repeat for all option combinations."""
        os.chdir(workspace_dir)  # Ensure we're in the workspace for plugin loading

        oop_plugins = load_plugins(["test_repeat"])
        legacy_plugins = load_plugins(["test_repeat_legacy"])

        for option in [None, "n_variants=3", "n_variants=2,suffix=-copy"]:
            option_map = {"test_repeat": option} if option else None
            oop_result = apply_plugin(*oop_plugins[0], "x", None, option_map)
            legacy_result = apply_plugin(*legacy_plugins[0], "x", None, {"test_repeat_legacy": option} if option else None)
            assert oop_result == legacy_result, \
                f"OOP and legacy results differ for option={option!r}: {oop_result} vs {legacy_result}"

    def test_repeat_custom_count_and_suffix(self, workspace_dir):
        """test_repeat n_variants=3 with custom suffix generates 3 correctly-named variants."""
        os.chdir(workspace_dir)  # Ensure we're in the workspace for plugin loading

        plugins = load_plugins(["test_repeat"])
        plugin_name, plugin_module = plugins[0]

        result = apply_plugin(plugin_name, plugin_module, "payload", None, {"test_repeat": "n_variants=3,suffix=-copy"})

        assert isinstance(result, list)
        assert len(result) == 3
        assert result == ["payload", "payload-copy", "payload-copy-2"]

    def test_piped_upper_then_base64(self, workspace_dir):
        """Piped test_upper|base64: 'hello' → 'HELLO' → 'SEVMTE8='"""
        os.chdir(workspace_dir)  # Ensure we're in the workspace for plugin loading

        plugins = load_plugins(["test_upper|base64"])
        plugin_name, plugin_modules = plugins[0]

        result = apply_plugin(plugin_name, plugin_modules, "hello", None, None)

        assert isinstance(result, list)
        assert "SEVMTE8=" in result

    def test_piped_upper_then_1337(self, workspace_dir):
        """Piped test_upper|1337: 'hello' → 'HELLO' → 'H3LL0' (E→3, O→0)"""
        os.chdir(workspace_dir)  # Ensure we're in the workspace for plugin loading

        plugins = load_plugins(["test_upper|1337"])
        plugin_name, plugin_modules = plugins[0]

        result = apply_plugin(plugin_name, plugin_modules, "hello", None, None)

        assert isinstance(result, list)
        assert "H3LL0" in result

    def test_piped_base64_then_1337(self, workspace_dir):
        """Piped base64|1337: 'hello' → 'aGVsbG8=' → '46V5868=' (a→4, G→6, s→5, b→8)"""
        os.chdir(workspace_dir)  # Ensure we're in the workspace for plugin loading

        plugins = load_plugins(["test_upper|1337|base64"])
        plugin_name, plugin_modules = plugins[0]

        result = apply_plugin(plugin_name, plugin_modules, "hello", None, None)

        assert isinstance(result, list)
        assert "SDNMTDA=" in result

    def test_exclude_patterns_token_preserved(self, workspace_dir):
        """1337 plugin with exclude_patterns leaves matched tokens verbatim while transforming the rest.
        
        "hello <SKIP> world" → "h3ll0 <SKIP> w0rld"
        """
        os.chdir(workspace_dir)  # Ensure we're in the workspace for plugin loading

        plugins = load_plugins(["1337"])
        plugin_name, plugin_module = plugins[0]

        result = apply_plugin(plugin_name, plugin_module, "hello <SKIP> world", ["<SKIP>"], None)

        assert isinstance(result, list)
        assert any("<SKIP>" in r for r in result), \
            f"Expected '<SKIP>' preserved verbatim in result, got: {result}"
        assert any("h3ll0" in r for r in result), \
            f"Expected 'hello' to be leet-transformed outside the excluded token, got: {result}"

    def test_multi_variant_plugin_mid_pipe_fans_out(self, workspace_dir):
        """A multi-variant plugin early in a pipe fans out: every variant is fed
        independently into the subsequent plugin.

        """
        os.chdir(workspace_dir)

        plugins = load_plugins(["test_repeat|base64"])
        plugin_name, plugin_modules = plugins[0]

        result = apply_plugin(plugin_name, plugin_modules, "payload", None, None)

        expected_plain = base64_lib.b64encode(b"payload").decode()
        expected_repeat = base64_lib.b64encode(b"payload-repeat").decode()

        assert isinstance(result, list)
        assert len(result) == 2, \
            f"Expected 2 variants (one per repeat output), got {len(result)}: {result}"
        assert expected_plain in result, \
            f"Expected base64('payload')='{expected_plain}' in result, got: {result}"
        assert expected_repeat in result, \
            f"Expected base64('payload-repeat')='{expected_repeat}' in result, got: {result}"