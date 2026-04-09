"""Test cases for the Entry class from spikee.generator."""

from spikee.generator import Entry, EntryType


class TestEntryInitialization:
    """Test Entry object initialization and basic properties."""

    def test_entry_initialization_document_type(self):
        """Test creating a basic DOCUMENT entry."""
        entry = Entry(
            entry_type=EntryType.DOCUMENT,
            entry_id="doc_001",
            base_id="base_001",
            jailbreak_id="jb_001",
            instruction_id="instr_001",
            prefix_id=None,
            suffix_id=None,
            text="This is a document",
            entry_text={},
            system_message=None,
            payload="jailbreak_text",
            lang="en",
            plugin_suffix="",
            plugin_name=None,
            judge_name="regex",
            judge_args="success",
            position="start",
            jailbreak_type="test",
            instruction_type="EN-CHECK",
            injection_pattern="INJECTION_PAYLOAD",
            spotlighting_data_markers=None,
        )
        
        assert entry.id == "doc_001"
        assert entry.base_id == "base_001"
        assert entry.lang == "en"
        assert entry.text == "This is a document"
        assert entry.entry_type == EntryType.DOCUMENT

    def test_entry_initialization_with_all_optional_fields(self):
        """Test creating an entry with all optional fields populated."""
        entry = Entry(
            entry_type=EntryType.DOCUMENT,
            entry_id="doc_002",
            base_id="base_002",
            jailbreak_id="jb_002",
            instruction_id="instr_002",
            prefix_id="prefix_123",
            suffix_id="suffix_456",
            text="Document with all fields",
            entry_text={},
            system_message="You are a helpful assistant",
            payload="full_jailbreak",
            lang="it",
            plugin_suffix="-plugin_upper",
            plugin_name="upper",
            judge_name="canary",
            judge_args="FLAG{secret}",
            position="middle",
            jailbreak_type="dev",
            instruction_type="restricted-check",
            injection_pattern="[INJECTION_PAYLOAD]",
            spotlighting_data_markers={"marker": "value"},
            exclude_from_transformations_regex=["pattern1", "pattern2"],
            steering_keywords=["keyword1", "keyword2"],
        )
        
        assert entry.prefix_id == "prefix_123"
        assert entry.suffix_id == "suffix_456"
        assert entry.system_message == "You are a helpful assistant"
        assert entry.plugin_name == "upper"
        assert entry.steering_keywords == ["keyword1", "keyword2"]


class TestEntryLongIdGeneration:
    """Test long_id generation for different entry types."""

    def test_long_id_document_entry(self):
        """Test long_id format for DOCUMENT entries."""
        entry = Entry(
            entry_type=EntryType.DOCUMENT,
            entry_id="doc_001",
            base_id="base_001",
            jailbreak_id="jb_001",
            instruction_id="instr_001",
            prefix_id=None,
            suffix_id=None,
            text="test",
            entry_text={},
            system_message=None,
            payload="payload",
            lang="en",
            plugin_suffix="",
            plugin_name=None,
            judge_name="regex",
            judge_args="success",
            position="start",
            jailbreak_type="test",
            instruction_type="EN-CHECK",
            injection_pattern="INJECTION_PAYLOAD",
            spotlighting_data_markers=None,
        )
        
        # long_id format: {entry_type}_{base_id}_{jailbreak_id}_{instruction_id}_{position}{plugin_suffix}
        assert entry.long_id == "document_base_001_jb_001_instr_001_start"

    def test_long_id_summary_entry(self):
        """Test long_id format for SUMMARY entries."""
        entry = Entry(
            entry_type=EntryType.SUMMARY,
            entry_id="summary_001",
            base_id="base_002",
            jailbreak_id="jb_002",
            instruction_id="instr_002",
            prefix_id=None,
            suffix_id=None,
            text="document text",
            entry_text={"ideal_summary": "summary"},
            system_message=None,
            payload="payload",
            lang="en",
            plugin_suffix="",
            plugin_name=None,
            judge_name="regex",
            judge_args="summary_check",
            position="end",
            jailbreak_type="test",
            instruction_type="EN-CHECK",
            injection_pattern="INJECTION_PAYLOAD",
            spotlighting_data_markers=None,
        )
        
        assert entry.long_id == "summarization_base_002_jb_002_instr_002_end"
        # SUMMARY entries should prepend "Summarize..." to text
        assert entry.text.startswith("Summarize the following document:")

    def test_long_id_qa_entry(self):
        """Test long_id format and text transformation for QA entries."""
        entry = Entry(
            entry_type=EntryType.QA,
            entry_id="qa_001",
            base_id="base_003",
            jailbreak_id="jb_003",
            instruction_id="instr_003",
            prefix_id=None,
            suffix_id=None,
            text="document text",
            entry_text={"question": "What is the answer?", "ideal_answer": "42"},
            system_message=None,
            payload="payload",
            lang="en",
            plugin_suffix="",
            plugin_name=None,
            judge_name="regex",
            judge_args="qa_check",
            position="middle",
            jailbreak_type="test",
            instruction_type="EN-CHECK",
            injection_pattern="INJECTION_PAYLOAD",
            spotlighting_data_markers=None,
        )
        
        assert entry.long_id == "qna_base_003_jb_003_instr_003_middle"
        # QA entries should include question in text
        assert "What is the answer?" in entry.text
        assert entry.text.startswith("Given this document:")

    def test_long_id_attack_entry(self):
        """Test long_id format for ATTACK entries."""
        entry = Entry(
            entry_type=EntryType.ATTACK,
            entry_id="attack_001",
            base_id="attack_base_123",
            jailbreak_id="jb_001",
            instruction_id="instr_001",
            prefix_id=None,
            suffix_id=None,
            text="attack text",
            entry_text={},
            system_message=None,
            payload="attack_payload",
            lang="en",
            plugin_suffix="-custom",
            plugin_name="custom",
            judge_name="regex",
            judge_args="attack_check",
            position="start",
            jailbreak_type="test",
            instruction_type="EN-CHECK",
            injection_pattern="INJECTION_PAYLOAD",
            spotlighting_data_markers=None,
        )
        
        # ATTACK entries have different long_id: {base_id}{plugin_suffix}
        assert entry.long_id == "attack_base_123-custom"

    def test_long_id_with_prefix_suffix_plugin(self):
        """Test long_id includes prefix, suffix, and system_message suffixes."""
        entry = Entry(
            entry_type=EntryType.DOCUMENT,
            entry_id="doc_004",
            base_id="base_004",
            jailbreak_id="jb_004",
            instruction_id="instr_004",
            prefix_id="001",
            suffix_id="002",
            text="test",
            entry_text={},
            system_message="system prompt",
            payload="payload",
            lang="en",
            plugin_suffix="-plugin_test",
            plugin_name="plugin_test",
            judge_name="regex",
            judge_args="test",
            position="start",
            jailbreak_type="test",
            instruction_type="EN-CHECK",
            injection_pattern="INJECTION_PAYLOAD",
            spotlighting_data_markers=None,
        )

        output = entry.to_entry()
        
        # long_id should include -p{prefix}, -s{suffix}, -sys suffixes
        assert "-p001" in output["long_id"]
        assert "-s002" in output["long_id"]
        assert "-sys" in output["long_id"]


class TestEntryToEntry:
    """Test the to_entry() method output."""

    def test_to_entry_basic_structure(self):
        """Test to_entry() returns correct basic structure."""
        entry = Entry(
            entry_type=EntryType.DOCUMENT,
            entry_id="doc_005",
            base_id="base_005",
            jailbreak_id="jb_005",
            instruction_id="instr_005",
            prefix_id=None,
            suffix_id=None,
            text="Test document",
            entry_text={},
            system_message=None,
            payload="test_payload",
            lang="en",
            plugin_suffix="",
            plugin_name=None,
            judge_name="regex",
            judge_args="test_arg",
            position="start",
            jailbreak_type="test",
            instruction_type="EN-CHECK",
            injection_pattern="INJECTION_PAYLOAD",
            spotlighting_data_markers=None,
        )
        
        output = entry.to_entry()
        
        # Check required fields
        assert output["id"] == "doc_005"
        assert output["long_id"] == entry.long_id
        assert output["text"] == "Test document"
        assert output["judge_name"] == "regex"
        assert output["judge_args"] == "test_arg"
        assert output["task_type"] == "document"
        assert output["jailbreak_type"] == "test"
        assert output["instruction_type"] == "EN-CHECK"
        assert output["document_id"] == "base_005"
        assert output["position"] == "start"
        assert output["lang"] == "en"
        assert output["plugin"] is None
        assert output["payload"] == "test_payload"
        assert output["injected"] == "true"

    def test_to_entry_with_plugin(self):
        """Test to_entry() includes plugin information."""
        entry = Entry(
            entry_type=EntryType.DOCUMENT,
            entry_id="doc_006",
            base_id="base_006",
            jailbreak_id="jb_006",
            instruction_id="instr_006",
            prefix_id=None,
            suffix_id=None,
            text="Transformed text",
            entry_text={},
            system_message=None,
            payload="payload",
            lang="en",
            plugin_suffix="-reverse",
            plugin_name="reverse",
            judge_name="regex",
            judge_args="test",
            position="start",
            jailbreak_type="test",
            instruction_type="EN-CHECK",
            injection_pattern="INJECTION_PAYLOAD",
            spotlighting_data_markers=None,
        )
        
        output = entry.to_entry()
        
        assert output["plugin"] == "reverse"

    def test_to_entry_summary_includes_ideal_summary(self):
        """Test to_entry() for SUMMARY includes ideal_summary."""
        entry = Entry(
            entry_type=EntryType.SUMMARY,
            entry_id="summary_002",
            base_id="base_007",
            jailbreak_id="jb_007",
            instruction_id="instr_007",
            prefix_id=None,
            suffix_id=None,
            text="long document",
            entry_text={"ideal_summary": "concise summary"},
            system_message=None,
            payload="payload",
            lang="en",
            plugin_suffix="",
            plugin_name=None,
            judge_name="regex",
            judge_args="test",
            position="start",
            jailbreak_type="test",
            instruction_type="EN-CHECK",
            injection_pattern="INJECTION_PAYLOAD",
            spotlighting_data_markers=None,
        )
        
        output = entry.to_entry()
        
        assert output["ideal_summary"] == "concise summary"

    def test_to_entry_qa_includes_ideal_answer(self):
        """Test to_entry() for QA includes ideal_answer."""
        entry = Entry(
            entry_type=EntryType.QA,
            entry_id="qa_002",
            base_id="base_008",
            jailbreak_id="jb_008",
            instruction_id="instr_008",
            prefix_id=None,
            suffix_id=None,
            text="document",
            entry_text={"question": "Q?", "ideal_answer": "Answer"},
            system_message=None,
            payload="payload",
            lang="en",
            plugin_suffix="",
            plugin_name=None,
            judge_name="regex",
            judge_args="test",
            position="start",
            jailbreak_type="test",
            instruction_type="EN-CHECK",
            injection_pattern="INJECTION_PAYLOAD",
            spotlighting_data_markers=None,
        )
        
        output = entry.to_entry()
        
        assert output["ideal_answer"] == "Answer"

    def test_to_entry_with_steering_keywords(self):
        """Test to_entry() includes steering keywords when present."""
        entry = Entry(
            entry_type=EntryType.DOCUMENT,
            entry_id="doc_009",
            base_id="base_009",
            jailbreak_id="jb_009",
            instruction_id="instr_009",
            prefix_id=None,
            suffix_id=None,
            text="test",
            entry_text={},
            system_message=None,
            payload="payload",
            lang="en",
            plugin_suffix="",
            plugin_name=None,
            judge_name="regex",
            judge_args="test",
            position="start",
            jailbreak_type="test",
            instruction_type="EN-CHECK",
            injection_pattern="INJECTION_PAYLOAD",
            spotlighting_data_markers=None,
            steering_keywords=["keyword1", "keyword2"],
        )
        
        output = entry.to_entry()
        
        assert output["steering_keywords"] == ["keyword1", "keyword2"]


class TestEntryToAttack:
    """Test the to_attack() method output."""

    def test_to_attack_basic_structure(self):
        """Test to_attack() returns correct structure."""
        entry = Entry(
            entry_type=EntryType.DOCUMENT,
            entry_id="doc_010",
            base_id="attack_base_010",
            jailbreak_id="jb_010",
            instruction_id="instr_010",
            prefix_id="p_010",
            suffix_id="s_010",
            text="Attack payload",
            entry_text={},
            system_message="attack system",
            payload="attack_payload",
            lang="en",
            plugin_suffix="-attack",
            plugin_name="attack_plugin",
            judge_name="regex",
            judge_args="attack_check",
            position="start",
            jailbreak_type="test",
            instruction_type="EN-CHECK",
            injection_pattern="INJECTION_PAYLOAD",
            spotlighting_data_markers=None,
        )
        
        output = entry.to_attack()
        
        # ATTACK format differs from entry format
        assert output["id"] == entry.long_id
        assert output["text"] == "Attack payload"
        assert output["judge_name"] == "regex"
        assert output["judge_args"] == "attack_check"
        # These should be None for attacks
        assert output["task_type"] is None
        assert output["document_id"] is None
        assert output["position"] is None
        assert output["spotlighting_data_markers"] is None
        assert output["injection_delimiters"] is None
        # Keep these
        assert output["payload"] == "attack_payload"
        assert output["plugin"] == "attack_plugin"
        assert output["prefix_id"] == "p_010"
        assert output["suffix_id"] == "s_010"

    def test_to_attack_with_steering_keywords(self):
        """Test to_attack() includes steering keywords."""
        entry = Entry(
            entry_type=EntryType.DOCUMENT,
            entry_id="doc_011",
            base_id="attack_011",
            jailbreak_id="jb_011",
            instruction_id="instr_011",
            prefix_id=None,
            suffix_id=None,
            text="Attack",
            entry_text={},
            system_message=None,
            payload="payload",
            lang="en",
            plugin_suffix="",
            plugin_name=None,
            judge_name="regex",
            judge_args="test",
            position="start",
            jailbreak_type="test",
            instruction_type="EN-CHECK",
            injection_pattern="INJECTION_PAYLOAD",
            spotlighting_data_markers=None,
            steering_keywords=["attack", "keyword"],
        )
        
        output = entry.to_attack()
        
        assert output["steering_keywords"] == ["attack", "keyword"]


class TestEntryEdgeCases:
    """Test edge cases and special scenarios."""

    def test_entry_with_empty_lang_defaults_to_en(self):
        """Test that empty lang defaults to 'en'."""
        entry = Entry(
            entry_type=EntryType.DOCUMENT,
            entry_id="doc_012",
            base_id="base_012",
            jailbreak_id="jb_012",
            instruction_id="instr_012",
            prefix_id=None,
            suffix_id=None,
            text="test",
            entry_text={},
            system_message=None,
            payload="payload",
            lang="",
            plugin_suffix="",
            plugin_name=None,
            judge_name="regex",
            judge_args="test",
            position="start",
            jailbreak_type="test",
            instruction_type="EN-CHECK",
            injection_pattern="INJECTION_PAYLOAD",
            spotlighting_data_markers=None,
        )
        
        assert entry.lang == "en"

    def test_entry_with_custom_lang(self):
        """Test entry with custom language."""
        entry = Entry(
            entry_type=EntryType.DOCUMENT,
            entry_id="doc_013",
            base_id="base_013",
            jailbreak_id="jb_013",
            instruction_id="instr_013",
            prefix_id=None,
            suffix_id=None,
            text="test",
            entry_text={},
            system_message=None,
            payload="payload",
            lang="fr",
            plugin_suffix="",
            plugin_name=None,
            judge_name="regex",
            judge_args="test",
            position="start",
            jailbreak_type="test",
            instruction_type="EN-CHECK",
            injection_pattern="INJECTION_PAYLOAD",
            spotlighting_data_markers=None,
        )
        
        assert entry.lang == "fr"

    def test_entry_qa_with_missing_question(self):
        """Test QA entry handles missing question gracefully."""
        entry = Entry(
            entry_type=EntryType.QA,
            entry_id="qa_003",
            base_id="base_014",
            jailbreak_id="jb_014",
            instruction_id="instr_014",
            prefix_id=None,
            suffix_id=None,
            text="document",
            entry_text={},  # Missing 'question' key
            system_message=None,
            payload="payload",
            lang="en",
            plugin_suffix="",
            plugin_name=None,
            judge_name="regex",
            judge_args="test",
            position="start",
            jailbreak_type="test",
            instruction_type="EN-CHECK",
            injection_pattern="INJECTION_PAYLOAD",
            spotlighting_data_markers=None,
        )
        
        # Should not raise error, but include empty string
        assert "Answer the following question:" in entry.text

    def test_entry_exclude_from_transformations_regex(self):
        """Test entry preserves exclude_from_transformations_regex."""
        patterns = ["regex1", "regex2", "regex3"]
        entry = Entry(
            entry_type=EntryType.DOCUMENT,
            entry_id="doc_015",
            base_id="base_015",
            jailbreak_id="jb_015",
            instruction_id="instr_015",
            prefix_id=None,
            suffix_id=None,
            text="test",
            entry_text={},
            system_message=None,
            payload="payload",
            lang="en",
            plugin_suffix="",
            plugin_name=None,
            judge_name="regex",
            judge_args="test",
            position="start",
            jailbreak_type="test",
            instruction_type="EN-CHECK",
            injection_pattern="INJECTION_PAYLOAD",
            spotlighting_data_markers=None,
            exclude_from_transformations_regex=patterns,
        )
        
        output = entry.to_entry()
        assert output["exclude_from_transformations_regex"] == patterns
