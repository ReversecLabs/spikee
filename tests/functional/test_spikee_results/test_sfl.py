import unittest

from spikee.utilities.sfl import SFLSyntaxError, matches_sfl, parse_sfl
from spikee.utilities.results import extract_entries, generate_query


class TestSFL(unittest.TestCase):
    def setUp(self):
        self.entry = {
            "success": True,
            "response": "Canary response",
            "error": None,
            "guardrail_categories": {"policy": "injection"},
            "meta": {"model": "Demo"},
            "attempts": 3,
            "tags": ["alpha", "beta"],
        }

    def test_boolean_operators_and_parentheses(self):
        query = '(success = true OR error) AND response LIKE "canary%"'
        self.assertTrue(matches_sfl(self.entry, parse_sfl(query)))

    def test_not_and_field_presence(self):
        self.assertTrue(matches_sfl(self.entry, parse_sfl("NOT error")))
        self.assertTrue(matches_sfl(self.entry, parse_sfl("guardrail_categories")))

    def test_full_entry_search_does_not_match_field_names(self):
        self.assertTrue(matches_sfl(self.entry, parse_sfl('"canary"')))
        self.assertFalse(matches_sfl(self.entry, parse_sfl('"guardrail_categories"')))

    def test_comparison_and_nested_path(self):
        self.assertTrue(matches_sfl(self.entry, parse_sfl('meta.model = "demo"')))
        self.assertTrue(matches_sfl(self.entry, parse_sfl('response != "refusal"')))
        self.assertFalse(matches_sfl(self.entry, parse_sfl("missing != null")))

    def test_like_uses_sql_wildcards(self):
        self.assertTrue(matches_sfl(self.entry, parse_sfl('response LIKE "canary%"')))
        self.assertTrue(matches_sfl(self.entry, parse_sfl('response LIKE "can_ry%"')))

    def test_has_matches_list_members(self):
        self.assertTrue(matches_sfl(self.entry, parse_sfl('tags HAS "ALPHA"')))
        self.assertFalse(matches_sfl(self.entry, parse_sfl('tags HAS "gamma"')))

    def test_numeric_comparisons(self):
        self.assertTrue(matches_sfl(self.entry, parse_sfl("attempts > 2")))
        self.assertTrue(matches_sfl(self.entry, parse_sfl("attempts >= 3")))
        self.assertTrue(matches_sfl(self.entry, parse_sfl("attempts < 4")))
        self.assertTrue(matches_sfl(self.entry, parse_sfl("attempts <= 3")))
        self.assertFalse(matches_sfl(self.entry, parse_sfl("response > 2")))

    def test_complex_filter(self):
        query = (
            '(success = true AND attempts >= 3) AND NOT error AND '
            '(response LIKE "%canary%" OR tags HAS "gamma") AND '
            'tags HAS "beta" AND meta.model = "demo"'
        )
        self.assertTrue(matches_sfl(self.entry, parse_sfl(query)))
        self.assertFalse(
            matches_sfl(self.entry, parse_sfl(query.replace('"beta"', '"gamma"')))
        )

    def test_invalid_syntax_reports_position(self):
        with self.assertRaisesRegex(SFLSyntaxError, "character"):
            parse_sfl("success AND OR error")
        with self.assertRaisesRegex(SFLSyntaxError, "use LIKE"):
            parse_sfl('response CONTAINS "canary"')

    def test_extraction_wrapper_is_sfl_only(self):
        query = generate_query("success = true")
        self.assertTrue(extract_entries(self.entry, query))


if __name__ == "__main__":
    unittest.main()