from collections import defaultdict
from jinja2 import Template
import html
from tabulate import tabulate
import os

from spikee.utilities.files import extract_resource_name, read_jsonl_file


# -- SPECIAL CHARACTER HANDLING --
def escape_special_chars(text):
    """Escapes special characters for console output."""
    if text is None:
        return "None"
    return repr(text)


def encode_special_characters(value):
    """Encodes special characters like newlines as '\\n' for Excel export."""
    if isinstance(value, str):
        return value.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return value  # If not a string, return as-is


def preprocess_results(results):
    """Preprocess results to encode special characters in specific fields."""
    for result in results:
        # Encode special characters in these fields if they exist
        if "injection_delimiters" in result:
            result["injection_delimiters"] = encode_special_characters(
                result["injection_delimiters"]
            )
        if "spotlighting_data_markers" in result:
            result["spotlighting_data_markers"] = encode_special_characters(
                result["spotlighting_data_markers"]
            )
    return results


# -- RESULT HELPERS --
def group_entries_with_attacks(results):
    """
    Group original entries with their corresponding dynamic attack entries.

    Returns:
        dict: A mapping from original IDs to a list of entries (original + attacks)
        dict: A mapping from attack IDs to their original ID
    """
    groups = defaultdict(list)
    attack_to_original = {}

    # First pass - identify all entries
    for entry in results:
        entry_id = entry["id"]
        source_file = entry.get(
            "source_file", None
        )  # Used for combined result analysis

        # Check if this is an attack entry
        if isinstance(entry_id, str) and "-attack" in entry_id:
            # Extract the original ID from the attack ID
            original_id_str = entry_id.split("-attack")[0]
            # Convert to the same type as the original ID (int or str)
            try:
                original_id = int(original_id_str)
            except ValueError:
                original_id = original_id_str

            if source_file is not None:  # For consistent key lookup
                str_original_id = str(original_id) + "-" + source_file
                str_entry_id = str(entry_id) + "-" + source_file
            else:
                str_original_id = str(original_id)
                str_entry_id = str(entry_id)

            groups[str_original_id].append(entry)
            attack_to_original[str_entry_id] = str_original_id
        else:
            # This is an original entry - use string representation for consistent keys
            if source_file is not None:
                str_entry_id = str(entry_id) + "-" + source_file
            else:
                str_entry_id = str(entry_id)
            groups[str_entry_id].append(entry)

    return groups, attack_to_original


def group_entries_by_source(results):
    """
    Group entries by their source file.

    Returns:
        dict: A mapping from source file names to a list of entries
    """
    source_groups = defaultdict(list)

    for entry in results:
        source_file = entry.get("source_file", "unknown_source")
        source_groups[source_file].append(entry)

    return source_groups


# -- RESULT PROCESSORS --
HTML_TEMPLATE = """
    <html>
    <head>
        <title>Results Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; }
            h1, h2, h3 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:hover { background-color: #f5f5f5; }
            pre { margin: 0; }
            .metrics { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .metric-good { color: green; }
            .metric-bad { color: red; }
            .attack-stats { background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>Results Analysis Report</h1>
        <p><strong>Result File:</strong> {{ result_file }}</p>

        <h2>General Statistics</h2>
        <ul>
            <li>Total Unique Entries: {{ total_entries }}</li>
            {% if has_dynamic_attacks %}
                <li>Successful Attacks (Total): {{ total_successes }}</li>
                <li>&nbsp;&nbsp;- Initially Successful: {{ initially_successful }}</li>
                <li>&nbsp;&nbsp;- Only Successful with Dynamic Attack: {{ attack_only_successful }}</li>
                <li>Failed Attacks: {{ total_failures }}</li>
                <li>Errors: {{ total_errors }}</li>
                <li>Total Attempts: {{ total_attempts }}</li>
                <li>Attack Success Rate (Overall): {{ attack_success_rate }}</li>
                <li>Attack Success Rate (Without Dynamic Attack): {{ "%.2f%%" | format(initial_success_rate) }}</li>
                <li>Attack Success Rate (Improvement from Dynamic Attack): {{ "%.2f%%" | format(attack_improvement) }}</li>
            {% else %}
                <li>Successful Attacks: {{ total_successes }}</li>
                <li>Failed Attacks: {{ total_failures }}</li>
                <li>Errors: {{ total_errors }}</li>
                <li>Total Attempts: {{ total_attempts }}</li>
                <li>Attack Success Rate: {{ attack_success_rate }}</li>
            {% endif %}
        </ul>

        {% if has_dynamic_attacks and attack_types %}
        <div class="attack-stats">
            <h3>Dynamic Attack Statistics</h3>
            <table>
                <tr>
                    <th>Attack Type</th>
                    <th>Total</th>
                    <th>Successes</th>
                    <th>Attempts</th>
                    <th>Success Rate</th>
                </tr>
                {% for attack_name, stats in attack_types.items() %}
                <tr>
                    <td>{{ attack_name }}</td>
                    <td>{{ stats.total }}</td>
                    <td>{{ stats.successes }}</td>
                    <td>{{ stats.attempts }}</td>
                    <td>{{ "%.2f%%" | format((stats.successes / stats.total * 100) if stats.total else 0) }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}

        {% if fp_data %}
        <h2>False Positive Analysis</h2>
        <p><strong>False Positive Check File:</strong> {{ fp_data.file }}</p>

        <div class="metrics">
            <h3>Confusion Matrix</h3>
            <ul>
                <li><strong>True Positives</strong> (attacks correctly blocked): {{ fp_data.true_positives }}</li>
                <li><strong>False Negatives</strong> (attacks incorrectly allowed): {{ fp_data.false_negatives }}</li>
                <li><strong>True Negatives</strong> (benign prompts correctly allowed): {{ fp_data.true_negatives }}</li>
                <li><strong>False Positives</strong> (benign prompts incorrectly blocked): {{ fp_data.false_positives }}</li>
            </ul>

            <h3>Performance Metrics</h3>
            <ul>
                <li><strong>Precision:</strong> {{ "%.4f"|format(fp_data.precision) }} - Of all blocked prompts, what fraction were actual attacks</li>
                <li><strong>Recall:</strong> {{ "%.4f"|format(fp_data.recall) }} - Of all actual attacks, what fraction were blocked</li>
                <li><strong>F1 Score:</strong> {{ "%.4f"|format(fp_data.f1_score) }} - Harmonic mean of precision and recall</li>
                <li><strong>Accuracy:</strong> {{ "%.4f"|format(fp_data.accuracy) }} - Overall accuracy across all prompts</li>
            </ul>
        </div>
        {% endif %}

        {% for field, breakdown in breakdowns.items() %}
            <h2>Breakdown by {{ field.replace('_', ' ').title() }}</h2>
            <table>
                <tr>
                    <th>{{ field.title() }}</th>
                    <th>Total</th>
                    <th>Successes</th>
                    {% if has_dynamic_attacks %}
                    <th>Initial Successes</th>
                    <th>Attack-Only Successes</th>
                    {% endif %}
                    <th>Attempts</th>
                    <th>Success Rate</th>
                    {% if has_dynamic_attacks %}
                    <th>Initial Success Rate</th>
                    <th>Attack Improvement</th>
                    {% endif %}
                </tr>
                {% for item in breakdown %}
                <tr>
                    <td><pre>{{ item.value }}</pre></td>
                    <td>{{ item.total }}</td>
                    <td>{{ item.successes }}</td>
                    {% if has_dynamic_attacks %}
                    <td>{{ item.initial_successes }}</td>
                    <td>{{ item.attack_only_successes }}</td>
                    {% endif %}
                    <td>{{ item.attempts }}</td>
                    <td>{{ item.success_rate }}</td>
                    {% if has_dynamic_attacks %}
                    <td>{{ item.initial_success_rate }}</td>
                    <td>{{ item.attack_improvement }}</td>
                    {% endif %}
                </tr>
                {% endfor %}
            </table>
        {% endfor %}

        <h2>Top 10 Most Successful Combinations</h2>
        <table>
            <tr>
                <th>Jailbreak Type</th>
                <th>Instruction Type</th>
                <th>Language</th>
                <th>Suffix ID</th>
                <th>Plugin</th>
                <th>Total</th>
                <th>Successes</th>
                {% if has_dynamic_attacks %}
                <th>Initial Successes</th>
                <th>Attack-Only Successes</th>
                {% endif %}
                <th>Attempts</th>
                <th>Success Rate</th>
                {% if has_dynamic_attacks %}
                <th>Initial Rate</th>
                <th>Attack Improvement</th>
                {% endif %}
            </tr>
            {% for combo in top_combinations %}
            <tr>
                <td>{{ combo.jailbreak_type }}</td>
                <td>{{ combo.instruction_type }}</td>
                <td>{{ combo.lang }}</td>
                <td>{{ combo.suffix_id }}</td>
                <td>{{ combo.plugin }}</td>
                <td>{{ combo.total }}</td>
                <td>{{ combo.successes }}</td>
                {% if has_dynamic_attacks %}
                <td>{{ combo.initial_successes }}</td>
                <td>{{ combo.attack_only_successes }}</td>
                {% endif %}
                <td>{{ combo.attempts }}</td>
                <td>{{ "%.2f%%" % combo.success_rate }}</td>
                {% if has_dynamic_attacks %}
                <td>{{ "%.2f%%" % combo.initial_success_rate }}</td>
                <td>{{ "%.2f%%" % combo.attack_improvement }}</td>
                {% endif %}
            </tr>
            {% endfor %}
        </table>

        <h2>Top 10 Least Successful Combinations</h2>
        <table>
            <tr>
                <th>Jailbreak Type</th>
                <th>Instruction Type</th>
                <th>Language</th>
                <th>Suffix ID</th>
                <th>Plugin</th>
                <th>Total</th>
                <th>Successes</th>
                {% if has_dynamic_attacks %}
                <th>Initial Successes</th>
                <th>Attack-Only Successes</th>
                {% endif %}
                <th>Attempts</th>
                <th>Success Rate</th>
                {% if has_dynamic_attacks %}
                <th>Initial Rate</th>
                <th>Attack Improvement</th>
                {% endif %}
            </tr>
            {% for combo in bottom_combinations %}
            <tr>
                <td>{{ combo.jailbreak_type }}</td>
                <td>{{ combo.instruction_type }}</td>
                <td>{{ combo.lang }}</td>
                <td>{{ combo.suffix_id }}</td>
                <td>{{ combo.plugin }}</td>
                <td>{{ combo.total }}</td>
                <td>{{ combo.successes }}</td>
                {% if has_dynamic_attacks %}
                <td>{{ combo.initial_successes }}</td>
                <td>{{ combo.attack_only_successes }}</td>
                {% endif %}
                <td>{{ combo.attempts }}</td>
                <td>{{ "%.2f%%" % combo.success_rate }}</td>
                {% if has_dynamic_attacks %}
                <td>{{ "%.2f%%" % combo.initial_success_rate }}</td>
                <td>{{ "%.2f%%" % combo.attack_improvement }}</td>
                {% endif %}
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """


class ResultProcessor:
    def __init__(self, results, result_file, fp_check_file=None):
        self.results = results
        self.result_file = result_file
        self.fp_check_file = fp_check_file

        self._fp_data = None
        self._combination_stats_sorted = []
        self._breakdowns = {}

        # --- 1. SETUP ---
        # Group original entries with their dynamic attack entries
        self._entry_groups, self._attack_to_original = group_entries_with_attacks(
            self.results
        )

        # Count unique entries (considering original + attack as one logical entry)
        self.total_entries = len(self._entry_groups)

        # Count success at the group level (success if any entry in the group succeeded)
        self.successful_groups = 0
        self.initially_successful_groups = 0  # Succeeded without dynamic attack
        self.attack_only_successful_groups = 0  # Succeeded only with dynamic attack
        self.failed_groups = 0
        self.guardrail_groups = 0
        self.error_groups = 0
        self.total_attempts = 0

        self.__init_calculations()

    def __init_calculations(self):
        """Initial calculations for statistics."""
        # Track which attack types were successful
        self.attack_types = defaultdict(
            lambda: {"total": 0, "successes": 0, "attempts": 0, "guardrail": 0}
        )

        # Track which unique features (like jailbreak_type) are associated with each group
        self.group_features = {}

        # Track Triggered Guardrail Categories
        self.guardrail_categories = {}

        # Check if any dynamic attacks were used
        self.has_dynamic_attacks = False

        # --- 2. PROCESS GROUPS ---
        for original_id, entries in self._entry_groups.items():
            # Count total attempts across all entries in the group
            group_attempts = sum(entry.get("attempts", 1) for entry in entries)
            self.total_attempts += group_attempts

            # Check if any entry in this group is a dynamic attack
            attack_entries = [
                e
                for e in entries
                if isinstance(e["id"], str) and "-attack" in str(e["id"])
            ]
            if attack_entries:
                self.has_dynamic_attacks = True

            # Check attack success (entries with -attack)
            attack_success = any(e.get("success", False) for e in attack_entries)

            # Check initial success (original entry without attack)
            initial_entries = [
                e
                for e in entries
                if not (isinstance(e["id"], str) and "-attack" in str(e["id"]))
            ]

            # Check if any of the initial entries were successful
            initial_success = any(e.get("success", False) for e in initial_entries)

            # Overall success
            group_success = initial_success or attack_success

            # check if all entries had guardrails triggered
            group_has_guardrail = all(
                entry.get("guardrail", False) for entry in entries
            )

            # Check if all entries had errors
            group_has_error = all(
                entry.get("error") not in [None, "No response received"]
                for entry in entries
            )

            # Track attack types
            for attack_entry in attack_entries:
                attack_name = attack_entry.get("attack_name", "None")
                if attack_name != "None":
                    # Clean up the attack name by removing 'spikee.' prefix
                    clean_attack_name = attack_name.replace(
                        "spikee.attacks.", ""
                    ).replace("spikee.", "")
                    self.attack_types[clean_attack_name]["total"] += 1
                    self.attack_types[clean_attack_name]["attempts"] += (
                        attack_entry.get("attempts", 1)
                    )
                    if attack_entry.get("success", False):
                        self.attack_types[clean_attack_name]["successes"] += 1

                    elif attack_entry.get("guardrail", False):
                        self.attack_types[clean_attack_name]["guardrail"] += 1

            # Increment appropriate counters
            if group_success:
                self.successful_groups += 1
                if initial_success:
                    self.initially_successful_groups += 1
                elif attack_success:
                    self.attack_only_successful_groups += 1
            elif group_has_error:
                self.error_groups += 1
                if group_has_guardrail:
                    self.guardrail_groups += 1
            else:
                self.failed_groups += 1

            # Increment guardrail categories
            for entry in entries:
                if entry.get("guardrail", False):
                    categories = entry.get("guardrail_categories", {})
                    for category, triggered in categories.items():
                        if triggered:
                            if category not in self.guardrail_categories:
                                self.guardrail_categories[category] = 0
                            self.guardrail_categories[category] += 1

            # Store the original entry's features to use in breakdowns
            # (We use the original entry for consistency)
            original_entry = next(
                (
                    e
                    for e in entries
                    if not isinstance(e["id"], str) or "-attack" not in str(e["id"])
                ),
                entries[0],
            )
            self.group_features[original_id] = original_entry

        # --- 3. CALCULATE STATISTICS ---
        # Calculate attack success rates
        self.attack_success_rate = (
            (self.successful_groups / self.total_entries) * 100
            if self.total_entries
            else 0
        )
        self.initial_success_rate = (
            (self.initially_successful_groups / self.total_entries) * 100
            if self.total_entries
            else 0
        )
        self.attack_improvement = (
            (self.attack_only_successful_groups / self.total_entries) * 100
            if self.total_entries
            else 0
        )

    def generate_output(self, overview=False, combined=False):
        """Generates the full results analysis output."""
        output = self.generate_overview()

        if combined:
            output += self.generate_combined()

        if not overview:
            output += self.generate_fp_analysis()
            output += self.generate_detailed_breakdowns()
            if len(self.guardrail_categories) > 0:
                output += self.generate_guardrail_categories()
            output += self.generate_combination_analysis()

        else:
            self._fp_data = None
            self._combination_stats_sorted = []

        return output

    def generate_html_report(self):
        """Generates an HTML report of the results analysis."""
        self.generate_fp_analysis()
        self.generate_detailed_breakdowns()
        if len(self.guardrail_categories) > 0:
            self.generate_guardrail_categories()
        self.generate_combination_analysis()

        # Prepare data for the template
        template_data = {
            "result_file": self.result_file,
            "total_entries": self.total_entries,
            "total_successes": self.successful_groups,
            "total_failures": self.failed_groups,
            "total_errors": self.error_groups,
            "total_attempts": self.total_attempts,
            "attack_success_rate": f"{self.attack_success_rate:.2f}%",
            "breakdowns": {},
            "top_combinations": self._combination_stats_sorted[:10],
            "bottom_combinations": [
                combo for combo in self._combination_stats_sorted if combo["total"] > 0
            ][-10:],
            "fp_data": self._fp_data,
            "initially_successful": self.initially_successful_groups,
            "attack_only_successful": self.attack_only_successful_groups,
            "initial_success_rate": self.initial_success_rate,
            "attack_improvement": self.attack_improvement,
            "attack_types": self.attack_types,
            "has_dynamic_attacks": self.has_dynamic_attacks,
        }

        # Prepare breakdown data
        for field, data in self._breakdowns.items():
            breakdown_list = []
            for value, stats in data.items():
                total = stats["total"]
                successes = stats["successes"]
                attempts = stats["attempts"]
                success_rate = (successes / total) * 100 if total else 0

                item = {
                    "value": html.escape(str(value)) if value else "None",
                    "total": total,
                    "successes": successes,
                    "attempts": attempts,
                    "success_rate": f"{success_rate:.2f}%",
                }

                # Add attack-specific stats if available
                if self.has_dynamic_attacks:
                    initial_successes = stats["initial_successes"]
                    attack_only_successes = stats["attack_only_successes"]
                    initial_success_rate = (
                        (initial_successes / total) * 100 if total else 0
                    )
                    attack_improvement = (
                        (attack_only_successes / total) * 100 if total else 0
                    )

                    item.update(
                        {
                            "initial_successes": initial_successes,
                            "attack_only_successes": attack_only_successes,
                            "initial_success_rate": f"{initial_success_rate:.2f}%",
                            "attack_improvement": f"{attack_improvement:.2f}%",
                        }
                    )

                # Replace newlines and tabs with visible representations
                item["value"] = item["value"].replace("\n", "\\n").replace("\t", "\\t")
                breakdown_list.append(item)

            # Sort by success rate descending
            breakdown_list.sort(
                key=lambda x: float(x["success_rate"].strip("%")), reverse=True
            )
            template_data["breakdowns"][field] = breakdown_list

        # Render template
        template = Template(HTML_TEMPLATE)
        html_content = template.render(template_data)

        # Write to HTML file
        output_file = os.path.splitext(self.result_file)[0] + "_analysis.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"HTML report generated: {output_file}")

    def generate_overview(self):
        """Generates the overview statistics section."""
        output = ""

        # General statistics
        output += f"\n=== Analysis of Results File: {self.result_file} ==="
        output += "\n=== General Statistics ==="
        output += f"\nTotal Unique Entries: {self.total_entries}"

        if self.has_dynamic_attacks:
            output += f"\nSuccessful Attacks (Total): {self.successful_groups} [{(self.successful_groups / self.total_entries) * 100:.2f}%]"
            output += f"\n  - Initially Successful: {self.initially_successful_groups} [{(self.initially_successful_groups / self.total_entries) * 100:.2f}%]"
            output += f"\n  - Only Successful with Dynamic Attack: {self.attack_only_successful_groups} [{(self.attack_only_successful_groups / self.total_entries) * 100:.2f}%]"
        else:
            output += f"\nSuccessful Attacks: {self.successful_groups} [{(self.successful_groups / self.total_entries) * 100:.2f}%]"

        output += f"\nFailed Attacks: {self.failed_groups} [{(self.failed_groups / self.total_entries) * 100:.2f}%]"
        output += f"\nErrors: {self.error_groups} [{(self.error_groups / self.total_entries) * 100:.2f}%]"

        if self.guardrail_groups > 0:
            output += f"\nGuardrail Triggers: {self.guardrail_groups} [{(self.guardrail_groups / self.total_entries) * 100:.2f}%]"
        output += f"\nTotal Attempts: {self.total_attempts}"
        output += f"\nAttack Success Rate (ASR): {self.attack_success_rate:.2f}%"

        # Dynamic attack statistics
        if self.attack_types:
            output += "\n\n=== Dynamic Attack Statistics ===\n"
            table = []
            for attack_name, stats in self.attack_types.items():
                success_rate = (
                    (stats["successes"] / stats["total"]) * 100 if stats["total"] else 0
                )
                table.append(
                    [
                        attack_name,
                        stats["total"],
                        stats["successes"],
                        stats["attempts"],
                        stats["guardrail"],
                        f"{success_rate:.2f}%",
                    ]
                )

            # Sort by success rate
            table.sort(key=lambda x: float(x[5].strip("%")), reverse=True)
            headers = [
                "Attack Type",
                "Total",
                "Successes",
                "Attempts",
                "Guardrail",
                "Success Rate",
            ]
            output += "\n" + tabulate(table, headers=headers) + "\n"

        return output

    def generate_combined(self):
        source_groups = group_entries_by_source(self.results)
        self._source_stats = {
            field: {
                "attempts": 0,
                "successes": 0,
                "initial_successes": 0,
                "attack_only_successes": 0,
                "failed": 0,
                "guardrail": 0,
                "errors": 0,
                "attack_success_rate": 0.0,
                "initial_success_rate": 0.0,
                "attack_improvement": 0.0,
            }
            for field in source_groups.keys()
        }

        for original_id, entries in self._entry_groups.items():
            source = entries[0].get("source_file", "unknown_source")
            self._source_stats[source]["attempts"] += sum(
                entry.get("attempts", 1) for entry in entries
            )

            attack_entries = [
                e
                for e in entries
                if isinstance(e["id"], str) and "-attack" in str(e["id"])
            ]
            attack_success = any(e.get("success", False) for e in attack_entries)

            initial_entries = [
                e
                for e in entries
                if not (isinstance(e["id"], str) and "-attack" in str(e["id"]))
            ]
            initial_success = any(e.get("success", False) for e in initial_entries)

            group_success = initial_success or attack_success
            attack_only_success = not initial_success and attack_success

            group_has_guardrail = all(
                entry.get("guardrail", False) for entry in entries
            )

            group_has_error = all(
                entry.get("error") not in [None, "No response received"]
                for entry in entries
            )

            if group_success:
                self._source_stats[source]["successes"] += 1
                if initial_success:
                    self._source_stats[source]["initial_successes"] += 1
                elif attack_only_success:
                    self._source_stats[source]["attack_only_successes"] += 1

            elif group_has_error:
                self._source_stats[source]["errors"] += 1
                if group_has_guardrail:
                    self._source_stats[source]["guardrail"] += 1

            else:
                self._source_stats[source]["failed"] += 1

        for source, stats in self._source_stats.items():
            total = stats["successes"] + stats["failed"] + stats["errors"]
            stats["attack_success_rate"] = (
                (stats["successes"] / total) * 100 if total else 0
            )
            stats["initial_success_rate"] = (
                (stats["initial_successes"] / total) * 100 if total else 0
            )
            stats["attack_improvement"] = (
                (stats["attack_only_successes"] / total) * 100 if total else 0
            )

        # Create a table of source stats sorted alphabetically by source name

        table = []
        headers = [
            "Source",
            "Attempts",
            "Successes",
            "Failed",
            "Guardrail",
            "Errors",
            "ASR",
        ]

        # Sort sources by attack success rate (ASR) descending
        for source, stats in sorted(
            self._source_stats.items(),
            key=lambda x: x[1]["attack_success_rate"],
            reverse=True,
        ):
            table.append(
                [
                    extract_resource_name(source),
                    stats["attempts"],
                    stats["successes"],
                    stats["failed"],
                    stats["guardrail"],
                    stats["errors"],
                    f"{stats['attack_success_rate']:.2f}%",
                ]
            )

        output = "\n\n=== Combined Source Statistics ===\n"
        output += tabulate(table, headers=headers) + "\n"

        return output

    def generate_fp_analysis(self):
        """Generates the false positive analysis section."""
        self._fp_data = None
        output = ""

        if self.fp_check_file:
            if not os.path.exists(self.fp_check_file):
                print(
                    f"\nWARNING: False positive check file '{self.fp_check_file}' not found or not accessible."
                )
            else:
                output += "\n=== False Positive Analysis ==="
                output += f"\nFalse Positive Check File: {self.fp_check_file}\n"

                try_output = ""
                try:
                    # Load the false positive checks data
                    fp_results = read_jsonl_file(self.fp_check_file)
                    try_output += f"\nLoaded {len(fp_results)} entries from false positive check file\n"

                    # Group the false positive entries with their dynamic attack entries (if any)
                    fp_groups, _ = group_entries_with_attacks(fp_results)
                    try_output += f"\nGrouped into {len(fp_groups)} unique entries\n"
                    # Count success/failure in false positive checks
                    fp_success = 0  # True Negatives (benign prompts correctly allowed)
                    fp_failure = (
                        0  # False Positives (benign prompts incorrectly blocked)
                    )

                    for fp_id, entries in fp_groups.items():
                        # If any entry in the group was successful, count it as a success
                        if any(entry.get("success", False) for entry in entries):
                            fp_success += 1
                        else:
                            fp_failure += 1

                    # Calculate confusion matrix values
                    false_negatives = (
                        self.successful_groups  # Attacks that went through (should have been blocked)
                    )
                    true_positives = (
                        self.failed_groups
                    )  # Attacks that were blocked (correctly)
                    true_negatives = (
                        fp_success  # Benign prompts that went through (correctly)
                    )
                    false_positives = (
                        fp_failure  # Benign prompts that were blocked (incorrectly)
                    )

                    # Calculate metrics
                    precision = (
                        true_positives / (true_positives + false_positives)
                        if (true_positives + false_positives) > 0
                        else 0
                    )
                    recall = (
                        true_positives / (true_positives + false_negatives)
                        if (true_positives + false_negatives) > 0
                        else 0
                    )
                    f1_score = (
                        2 * (precision * recall) / (precision + recall)
                        if (precision + recall) > 0
                        else 0
                    )
                    accuracy = (
                        (true_positives + true_negatives)
                        / (
                            true_positives
                            + true_negatives
                            + false_positives
                            + false_negatives
                        )
                        if (
                            true_positives
                            + true_negatives
                            + false_positives
                            + false_negatives
                        )
                        > 0
                        else 0
                    )

                    # Confusion Matrix and Performance Metrics
                    try_output += f"""
=== Confusion Matrix ===
True Positives (attacks correctly blocked): {true_positives}
False Negatives (attacks incorrectly allowed): {false_negatives}
True Negatives (benign prompts correctly allowed): {true_negatives}
False Positives (benign prompts incorrectly blocked): {false_positives}

=== Performance Metrics ===
Precision: {precision:.4f} - Of all blocked prompts, what fraction were actual attacks
Recall: {recall:.4f} - Of all actual attacks, what fraction were blocked
F1 Score: {f1_score:.4f} - Harmonic mean of precision and recall
Accuracy: {accuracy:.4f} - Overall accuracy across all prompts
"""
                    self._fp_data = {
                        "file": self.fp_check_file,
                        "true_positives": true_positives,
                        "false_negatives": false_negatives,
                        "true_negatives": true_negatives,
                        "false_positives": false_positives,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1_score,
                        "accuracy": accuracy,
                    }

                except Exception as e:
                    try_output = f"\nError processing false positive check file: {e}\n"

                output += try_output
        else:
            output += "\nNo false positive check file provided.\n"
        return output

    def generate_detailed_breakdowns(self):
        """Generates detailed breakdowns by various features."""
        # Initialize counters for breakdowns
        breakdown_fields = [
            "jailbreak_type",
            "instruction_type",
            "task_type",
            "position",
            "spotlighting_data_markers",
            "injection_delimiters",
            "lang",
            "suffix_id",
            "plugin",
        ]

        self._breakdowns = {
            field: defaultdict(
                lambda: {
                    "total": 0,
                    "successes": 0,
                    "initial_successes": 0,
                    "attack_only_successes": 0,
                    "attempts": 0,
                }
            )
            for field in breakdown_fields
        }

        # Initialize combination counters
        self._combination_counts = defaultdict(
            lambda: {
                "total": 0,
                "successes": 0,
                "initial_successes": 0,
                "attack_only_successes": 0,
                "attempts": 0,
            }
        )

        # Process groups for breakdowns
        for original_id, entry in self.group_features.items():
            # Get all entries in this group
            entries = self._entry_groups[original_id]

            # Check success types
            initial_entries = [
                e
                for e in entries
                if not (isinstance(e["id"], str) and "-attack" in str(e["id"]))
            ]
            attack_entries = [
                e
                for e in entries
                if isinstance(e["id"], str) and "-attack" in str(e["id"])
            ]

            initial_success = any(e.get("success", False) for e in initial_entries)
            attack_success = any(e.get("success", False) for e in attack_entries)
            overall_success = initial_success or attack_success
            attack_only_success = not initial_success and attack_success

            # Total attempts for this group
            group_attempts = sum(e.get("attempts", 1) for e in entries)

            # Prepare fields, replacing missing or empty values with 'None'
            jailbreak_type = entry.get("jailbreak_type") or "None"
            instruction_type = entry.get("instruction_type") or "None"
            lang = entry.get("lang") or "None"
            suffix_id = entry.get("suffix_id") or "None"
            plugin = entry.get("plugin") or "None"

            # Update combination counts
            combo_key = (jailbreak_type, instruction_type, lang, suffix_id, plugin)
            self._combination_counts[combo_key]["total"] += 1
            self._combination_counts[combo_key]["attempts"] += group_attempts
            if overall_success:
                self._combination_counts[combo_key]["successes"] += 1
            if initial_success:
                self._combination_counts[combo_key]["initial_successes"] += 1
            if attack_only_success:
                self._combination_counts[combo_key]["attack_only_successes"] += 1

            # Update breakdowns
            for field in breakdown_fields:
                value = entry.get(field, "None") or "None"
                self._breakdowns[field][value]["total"] += 1
                self._breakdowns[field][value]["attempts"] += group_attempts
                if overall_success:
                    self._breakdowns[field][value]["successes"] += 1
                if initial_success:
                    self._breakdowns[field][value]["initial_successes"] += 1
                if attack_only_success:
                    self._breakdowns[field][value]["attack_only_successes"] += 1

        # Output breakdowns
        output = ""
        for field in breakdown_fields:
            data = self._breakdowns[field]

            if data:
                output += f"\n=== Breakdown by {field.replace('_', ' ').title()} ==="
                table = []

                if self.has_dynamic_attacks:
                    # Full version with attack statistics
                    for value, stats in data.items():
                        total = stats["total"]
                        successes = stats["successes"]
                        initial_successes = stats["initial_successes"]
                        attack_only_successes = stats["attack_only_successes"]
                        attempts = stats["attempts"]

                        success_rate = (successes / total) * 100 if total else 0
                        initial_success_rate = (
                            (initial_successes / total) * 100 if total else 0
                        )
                        attack_improvement = (
                            (attack_only_successes / total) * 100 if total else 0
                        )

                        escaped_value = escape_special_chars(value)
                        table.append(
                            [
                                escaped_value,
                                total,
                                successes,
                                initial_successes,
                                attack_only_successes,
                                attempts,
                                f"{success_rate:.2f}%",
                                f"{initial_success_rate:.2f}%",
                                f"{attack_improvement:.2f}%",
                            ]
                        )

                    # Sort the table by overall success rate descending
                    table.sort(key=lambda x: float(x[6].strip("%")), reverse=True)
                    headers = [
                        field.title(),
                        "Total",
                        "All Successes",
                        "Initial Successes",
                        "Attack-Only Successes",
                        "Attempts",
                        "Success Rate",
                        "Initial Success Rate",
                        "Attack Improvement",
                    ]
                else:
                    # Simplified version without attack statistics
                    for value, stats in data.items():
                        total = stats["total"]
                        successes = stats["successes"]
                        attempts = stats["attempts"]

                        success_rate = (successes / total) * 100 if total else 0

                        escaped_value = escape_special_chars(value)
                        table.append(
                            [
                                escaped_value,
                                total,
                                successes,
                                attempts,
                                f"{success_rate:.2f}%",
                            ]
                        )

                    # Sort the table by success rate descending
                    table.sort(key=lambda x: float(x[4].strip("%")), reverse=True)
                    headers = [
                        field.title(),
                        "Total",
                        "Successes",
                        "Attempts",
                        "Success Rate",
                    ]

                output += "\n" + tabulate(table, headers=headers) + "\n"

        return output

    def generate_guardrail_categories(self):
        """Generates the guardrail category breakdown section."""
        output = ""
        output += "\n=== Guardrail Category Breakdown ==="
        table = []
        for category, count in self.guardrail_categories.items():
            table.append(
                [category, count, f"{(count / self.total_entries) * 100:.2f}%"]
            )

        # Sort by count descending
        table.sort(key=lambda x: x[1], reverse=True)
        headers = ["Guardrail Category", "Trigger Count", "Trigger Rate"]
        output += "\n" + tabulate(table, headers=headers) + "\n"
        return output

    def generate_combination_analysis(self):
        """Generates the combination analysis section."""
        output = ""

        # Analyze combinations
        combination_stats = []
        for combo, stats in self._combination_counts.items():
            total = stats["total"]
            successes = stats["successes"]
            initial_successes = stats["initial_successes"]
            attack_only_successes = stats["attack_only_successes"]
            attempts = stats["attempts"]

            success_rate = (successes / total) * 100 if total else 0
            initial_success_rate = (initial_successes / total) * 100 if total else 0
            attack_improvement = (attack_only_successes / total) * 100 if total else 0

            combination_stats.append(
                {
                    "jailbreak_type": combo[0],
                    "instruction_type": combo[1],
                    "lang": combo[2],
                    "suffix_id": combo[3],
                    "plugin": combo[4],
                    "total": total,
                    "successes": successes,
                    "initial_successes": initial_successes,
                    "attack_only_successes": attack_only_successes,
                    "attempts": attempts,
                    "success_rate": success_rate,
                    "initial_success_rate": initial_success_rate,
                    "attack_improvement": attack_improvement,
                }
            )

        # Sort combinations by success rate
        self._combination_stats_sorted = sorted(
            combination_stats, key=lambda x: x["success_rate"], reverse=True
        )

        # Get top 10 most successful combinations
        top_10 = self._combination_stats_sorted[:10]

        # Get bottom 10 least successful combinations (excluding combinations with zero total)
        bottom_10 = [
            combo for combo in self._combination_stats_sorted if combo["total"] > 0
        ][-10:]

        # Get top 10 and bottom 10 combinations
        output += self.generate_combination_stats(
            "Top 10 Most Successful Combinations", top_10
        )
        output += self.generate_combination_stats(
            "Top 10 Least Successful Combinations", bottom_10
        )
        return output

    def generate_combination_stats(self, title, combo_list):
        """Helper function to print combination statistics."""
        output = f"\n=== {title} ==="
        table = []

        if self.has_dynamic_attacks:
            # Full version with attack statistics
            for combo in combo_list:
                jailbreak_type = escape_special_chars(combo["jailbreak_type"])
                instruction_type = escape_special_chars(combo["instruction_type"])
                lang = escape_special_chars(combo["lang"])
                suffix_id = escape_special_chars(combo["suffix_id"])
                plugin = escape_special_chars(combo["plugin"])

                total = combo["total"]
                successes = combo["successes"]
                initial_successes = combo["initial_successes"]
                attack_only_successes = combo["attack_only_successes"]
                attempts = combo["attempts"]

                success_rate = f"{combo['success_rate']:.2f}%"
                initial_success_rate = f"{combo['initial_success_rate']:.2f}%"
                attack_improvement = f"{combo['attack_improvement']:.2f}%"

                table.append(
                    [
                        jailbreak_type,
                        instruction_type,
                        lang,
                        suffix_id,
                        plugin,
                        total,
                        successes,
                        initial_successes,
                        attack_only_successes,
                        attempts,
                        success_rate,
                        initial_success_rate,
                        attack_improvement,
                    ]
                )

            headers = [
                "Jailbreak Type",
                "Instruction Type",
                "Language",
                "Suffix ID",
                "Plugin",
                "Total",
                "All Successes",
                "Initial Successes",
                "Attack-Only Successes",
                "Attempts",
                "Success Rate",
                "Initial Rate",
                "Attack Improvement",
            ]
        else:
            # Simplified version without attack statistics
            for combo in combo_list:
                jailbreak_type = escape_special_chars(combo["jailbreak_type"])
                instruction_type = escape_special_chars(combo["instruction_type"])
                lang = escape_special_chars(combo["lang"])
                suffix_id = escape_special_chars(combo["suffix_id"])
                plugin = escape_special_chars(combo["plugin"])

                total = combo["total"]
                successes = combo["successes"]
                attempts = combo["attempts"]

                success_rate = f"{combo['success_rate']:.2f}%"

                table.append(
                    [
                        jailbreak_type,
                        instruction_type,
                        lang,
                        suffix_id,
                        plugin,
                        total,
                        successes,
                        attempts,
                        success_rate,
                    ]
                )

            headers = [
                "Jailbreak Type",
                "Instruction Type",
                "Language",
                "Suffix ID",
                "Plugin",
                "Total",
                "Successes",
                "Attempts",
                "Success Rate",
            ]

        output += "\n" + tabulate(table, headers=headers) + "\n"
        return output


# -- EXTRACT --
def generate_query(category, custom_search=None):
    """Generates a search query based on the specified category or custom search criteria."""

    # Category validation
    if category not in [
        "success",
        "failure",
        "error",
        "guardrail",
        "no-guardrail",
        "custom",
    ]:
        raise ValueError("Invalid category specified for extraction.")

    custom_query = []
    if category == "custom":
        if custom_search is None:
            raise ValueError(
                "Custom search query must be provided for 'custom' category."
            )

        for query in custom_search:
            query = query.split(":", 1)
            query.reverse()
            custom_query.append(query)

    return custom_query


def extract_entries(entry, category="success", custom_query=None):
    """Extracts entries based on the specified category or custom search criteria."""
    match category:
        case "success":
            if entry.get("success", False):
                return True

        case "failure":
            if not entry.get("success", False):
                return True

        case "error":
            if entry.get("error") not in [None, "No response received"]:
                return True

        case "guardrail":
            if entry.get("guardrail", False):
                return True

        case "no-guardrail":
            if not entry.get("guardrail", False):
                return True

        case "custom":
            query_match = True
            for query in custom_query:
                if len(query) > 1:
                    query, field = query[0], query[1]

                    if not extract_search(entry, query, field):
                        query_match = False

                elif not extract_search(entry, query[0]):
                    query_match = False

            return query_match

    return False


def extract_search(entry, query: str, field: str = None):
    """Searches for a query in the given text, supporting inversion with '!' prefix."""

    try:
        q_invert = query.startswith("!")
        if q_invert:
            query = query[1:]

        if field is not None:
            f_invert = field.startswith("!")
            if f_invert:
                field = field[1:]

            text = entry.get(field, None)
            if text is None:
                return f_invert

            text = str(text)

        else:
            f_invert = False
            text = str(entry)

        result = query in text
        return not result if q_invert else result

    except Exception as e:
        print(f"Error during search extraction (Entry {entry}): {e}")
        return False
