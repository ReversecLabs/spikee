# Spikee Results

The `spikee results` command includes several tools to aid in the analysis of test results.
* `spikee results analyze`: Processes results files to provide a detailed breakdown of performance statistics.
* `spikee results rejudge`: See [re-judging](<./08_judges.md#Re-judging>), revaluate results using a different judge or options.
* `spikee results extract`: Extracts SFL-matched results from result files for further analysis.
* `spikee results dataset-comparison`: Compares the results of a single dataset across multiple targets to identify trends or differences in performance.
* `spikee results convert-to-excel`: Converts results files into Excel format.


# The `analyze` Command
## Usage Examples
```bash
# Analyze a single results file, providing all standard metrics and breakdowns
spikee results analyze --result-file results/results_llm_provider-openai_gpt-4o-mini_*.jsonl
```

### Multi-file Options
Spikee allows for multiple results files to be provided in a single analysis run using the `--result-file` and `--result-folder` flags.
* Results files processed sequentially in the provided order (folders are scanned and included in alphabetical order), combining all results into a single analysis.
* The `--result-folder` flag will scan the specified folder for all JSONL files starting with `results_`.
* Multi-file is not compatible with single dataset flags (e.g., `--false-positive-checks`).
* By default, when multiple results files are included in an analysis, each results file will be processed and outputted separately. To combine all results into a single analysis, use the `--combine` flag.

```bash
# Analyze multiple results files together.
spikee results analyze --result-file results/results_llm_provider_cybersec-2026-01-*.jsonl \
                       --result-file results/results_llm_provider_simsonsun-high-quality-jailbreaks-*.jsonl
```

```bash
# Analyze all results files in a folder together.
spikee results analyze --result-folder results/
```

### Analysis Options
The command supports several options to customize the output of the analysis:
* `--overview`: Only output the general statistics of an analysis.
* `--combine`: Combine multiple results files into a single analysis.
* `--false-positive-checks <benign_results_file>`: Performs false positive analysis using a benign results file.

## Analysis Output Explained
### General Statistics / Overview

These top-level statistics provide a high-level summary of the test run:
*   **Total Unique Entries:** The number of distinct test cases processed. Each original prompt from your dataset is considered one unique entry, even if it triggers a dynamic attack with multiple iterations. This is the denominator for the overall success rate.
*   **Successful Attacks (Total):** The number of unique entries where at least one attempt (either a standard prompt or a dynamic attack iteration) was deemed successful by its Judge.
*   **Failed Attacks:** The number of unique entries where *all* attempts failed and no errors occurred.
*   **Errors:** The number of unique entries where *all* attempts resulted in an error (e.g., an API timeout or connection failure). An entry is only counted as an error if none of its attempts succeeded or failed normally.
*   **Guardrail Triggered:** The number of unique entries where *all* attempts were blocked by a guardrail (if applicable). An entry is only counted here if none of its attempts succeeded, failed normally, or errored.
*   **Total Attempts:** The sum of top-level `attempts` counts, including standard attempts and the iteration counts returned by dynamic attacks. These counts are not an exact transport-request or API-cost audit: an attack can count a failed generation step, and transport retries need not appear as separate candidates.
*   **Attack Success Rate (Overall):** `(Successful Attacks / Total Unique Entries) * 100`. This is the main metric for assessing the target's overall vulnerability during the test.

### Dynamic Attack Metrics

These metrics appear only if a dynamic attack was run using the `--attack` flag. They help you distinguish between vulnerabilities found by your initial dataset and those found by an adaptive attack script.

*   **Initially Successful:** The number of entries that succeeded using the standard prompt, *before* any dynamic attack was needed. This measures the baseline vulnerability to the static payloads in your dataset.
*   **Only Successful with Dynamic Attack:** The number of entries that succeeded *only* because the dynamic attack script found a working variation. This quantifies the additional risk uncovered by the attack strategy.
*   **Attack Success Rate (Improvement from Dynamic Attack):** The percentage of total entries that were only broken by the dynamic attack. This directly measures the value added by the adaptive attack logic.
*   **Dynamic Attack Statistics Table:** This table breaks down the performance of each dynamic attack script used:
    *   `Attack Type`: The name of the attack script.
    *   `Total`: How many times this dynamic attack script was *invoked* (i.e., for how many entries).
    *   `Successes`: How many times this script ultimately achieved a success.
    *   `Guardrail Triggers`: How many times this script was blocked by a guardrail (if applicable).
    *   `Success Rate`: `(Successes / Total) * 100`. This shows how effective a specific attack strategy was *when it was used*.

### False Positive Analysis Metrics (For Guardrails)

These metrics evaluate how well a guardrail target distinguishes between malicious attacks and benign prompts. This analysis requires two test runs (one on an attack dataset, one on a benign dataset) and using the `--false-positive-checks` flag. Remember the logic for a guardrail target: `True` means bypassed/allowed, `False` means blocked.

*   **Confusion Matrix:**
    *   **True Positives (TP):** Malicious attacks that were correctly **blocked**. (Attack run result is `False`).
    *   **False Negatives (FN):** Malicious attacks that were incorrectly **allowed** (bypassed). (Attack run result is `True`). These are successful attacks.
    *   **True Negatives (TN):** Benign prompts that were correctly **allowed**. (Benign run result is `True`).
    *   **False Positives (FP):** Benign prompts that were incorrectly **blocked**. (Benign run result is `False`).
*   **Performance Metrics:**
    *   **Precision:** `TP / (TP + FP)`. Of all prompts the guardrail *blocked*, what fraction were actual attacks? High precision means the guardrail does not over-block legitimate traffic.
    *   **Recall (Sensitivity):** `TP / (TP + FN)`. Of all *actual attacks* presented, what fraction did the guardrail correctly block? High recall means the guardrail is effective at catching threats.
    *   **F1 Score:** `2 * (Precision * Recall) / (Precision + Recall)`. The harmonic mean of Precision and Recall, providing a single score that balances both metrics.
    *   **Accuracy:** `(TP + TN) / (TP + TN + FP + FN)`. The overall percentage of all prompts (both attack and benign) that were classified correctly.

### Breakdown and Combination Tables

These tables help you drill down into the results to identify specific patterns of vulnerability. They break down success rates by factors like `Jailbreak Type`, `Instruction Type`, `Plugin`, and `Language`.

By sorting these tables (they are sorted by success rate by default), you can quickly answer questions like:
*   Which type of jailbreak is most effective against my target?
*   Is my target more vulnerable to XSS-related instructions or data exfiltration?
*   Does Base64 encoding (via a plugin) increase or decrease the success rate?

### Actionable Insights and Limitations

The results from `spikee` can help you answer key security questions, but it's important to understand their context.

**Questions You Can Answer:**
*   **Overall Resilience:** What is the baseline success rate of prompt injection attacks against my application using a specific dataset?
*   **Specific Vulnerabilities:** Is my application susceptible to a specific class of attack, like data exfiltration using Markdown, as represented in the test set?
*   **Guardrail Efficacy:** How effective is my guardrail at blocking known attacks (Recall), and how much does it interfere with legitimate use (Precision)?
*   **Regression Testing:** Did a recent change to my application's prompt or model introduce a new vulnerability that was previously caught?

**Limitations and Caveats:**
*   **Dataset-Dependent Results:** The results are entirely dependent on the dataset used. A 0% success rate means the target was resilient to the *specific attacks in your dataset*, not that it is immune to all possible attacks. The space of potential jailbreaks and prompt injections is vast, and it is not feasible to create a dataset that covers all possibilities.
*   **Severity Not Measured:** The success rate does not measure the *impact* or *severity* of a successful attack. A successful "data exfiltration" attack is likely more severe than a successful "make a joke" attack, but both count as one success in the statistics. Manual review of successful attacks in the `results.jsonl` file is essential to understand the true risk.

# The `extract` Command
This command extracts results from one or more results files using a required Spikee Filter Language (SFL) query.

## Usage Examples
```bash
# Extract all successful attacks to a new dataset file for further analysis
spikee results extract --result-file results/results_llm_provider_cybersec-2026-01-*.jsonl \
                       --result-file results/results_llm_provider_simsonsun-high-quality-jailbreaks-*.jsonl \
                       --query "success = true"
```

```bash
# Extract successful results whose response includes a canary
spikee results extract --result-file results/results_llm_provider_cybersec-2026-01-*.jsonl \
                       --result-file results/results_llm_provider_simsonsun-high-quality-jailbreaks-*.jsonl \
                       --query "success = true AND response LIKE \"%canary%\""
```

### Spikee Filter Language (SFL)
The `extract` command requires one SFL expression through `--query`. SFL is also used by the Results page in the WebUI. It replaces the earlier extraction categories, `field:value`, `!term`, whitespace-AND, and pipe-OR search syntax.

Use `AND`, `OR`, `NOT`, and parentheses to combine conditions. Operators are case-insensitive; `NOT` binds first, followed by `AND`, then `OR`.

| Query | Matches |
| --- | --- |
| `success = true` | Successful entries |
| `success = false` | Failed entries |
| `error AND error != "No response received"` | Entries with a recorded error |
| `guardrail = true` | Entries blocked by a guardrail |
| `guardrail = false OR NOT guardrail EXISTS` | Entries not blocked by a guardrail |
| `response != "I cannot respond to that"` | Entries with a different response |
| `response LIKE "%canary%"` | Case-insensitive SQL-style wildcard match; `%` is any sequence and `_` is one character |
| `attempts >= 3` | Numeric comparison; supports `<`, `<=`, `>`, and `>=` |
| `tags HAS "canary"` | Case-insensitive membership in a list, tuple, or set |
| `guardrail_categories` | Entries where the field exists and is not `null` |
| `NOT error` | Entries without an error field value |
| `meta.model = "gpt-4o"` | A nested dictionary field, addressed with a dot path |
| `"refusal" OR guardrail` | Text anywhere in entry values, or a non-null `guardrail` field |

Unquoted `true`, `false`, `null`, and numbers are typed literals. Quoted values are case-insensitive strings. A standalone quoted string searches all entry values but not field names. Missing fields never match comparisons, including `!=`. Use `LIKE` with `%` wildcards for string matching; `CONTAINS` is not an SFL operator.

# The `dataset-comparison` Command
This command is used to compare the results of a single dataset across multiple targets. It helps identify trends or differences in performance between different target configurations.

## Usage Examples
```bash
# Compares the results of three different targets (Target A, B and C) against the same dataset, identifying which entries had a success rate greater than then threshold of 80%.
spikee results dataset-comparison --dataset-file datasets/cybersec-2026-01-*.jsonl \
                                  --result-file results/results_target-A-cybersec-2026-01-*.jsonl \
                                  --result-file results/results_target-B-cybersec-2026-01-*.jsonl \
                                  --result-file results/results_target-C-cybersec-2026-01-*.jsonl \
                                  --success-threshold 0.8 \
                                  --success-definition gt
```

```bash
# Compares the results of three different targets (Target A, B and C) against the same dataset, identifying which entries had a success rate less than the threshold of 20%, outputting the top 10 entries.
spikee results dataset-comparison --dataset-file datasets/cybersec-2026-01-*.jsonl \
                                  --result-file results/results_target-A-cybersec-2026-01-*.jsonl \
                                  --result-file results/results_target-B-cybersec-2026-01-*.jsonl \
                                  --result-file results/results_target-C-cybersec-2026-01-*.jsonl \
                                  --success-threshold 0.2 \
                                  --success-definition lt
                                  --number 10
```

# The `spikee webui` Command
Launches the Spikee web UI — a browser-based interface for the full Spikee workflow. Run it from a workspace directory (one initialised with `spikee init`).

The web UI covers:
- **Generate** — configure and launch dataset generation jobs
- **Test** — configure and launch test runs against targets
- **Jobs** — monitor running jobs with live log output, cancel or re-run them
- **Results** — explore, filter, and bulk-action test results interactively

## Usage Examples
```bash
# Launch on the default host and port (127.0.0.1:8080)
spikee webui

# Bind to all interfaces on a custom port
spikee webui --host 0.0.0.0 -p 8081

# Persist job history to a SQLite database across restarts
spikee webui --database jobs.db
```
## Attack history and conversation results

Dynamic attacks write one result with ID `<id>-attack`. Its top-level input/response is the successful or final failed candidate, `success` is the overall outcome, and `attempts` is the total iteration count. The attack can supply additional evidence in that same row:

| Field | Contents | Typical attacks |
|---|---|---|
| `attempt_history` | A list of candidate inputs, responses, recorded verdicts, and optional errors | `best_of_n`, `llm_jailbreaker`, `llm_multi_language_jailbreaker` |
| `conversation` | The existing conversation graph or recorded message sequence | Crescendo, GOAT, Echo Chamber, `multi_turn` |

Supporting single-turn attacks collect history by default. `SPIKEE_ATTACK_HISTORY=false` disables that optional collection; it does not affect conversation logs. Legacy modules and other custom attacks may return neither field. A missing history does not imply one iteration, and `attempts=20` does not guarantee twenty saved candidates or twenty independent conversations.

For example, a result with `attempts=20` and twenty `attempt_history` items contributes **20 dynamic attempts and one attacked dataset entry**. Count top-level `attempts` once; do not add nested items or graph nodes. A separate standard attempt adds its own count. Success rates use the overall result verdict and dataset-parent identity, not the fraction of nested candidates marked successful.

Each nested history item has `input`, `response`, and `success`, with optional `error`. `success=null` means unjudged, not failed. Stored items also have a 1-based `invocation` identifying the outer `--attempts` call that supplied them. The runner combines provided candidate histories across those calls. Conversation graphs remain from the successful or final invocation and may omit details on existing error paths; candidate history does not extend their logging.

The viewer displays candidate history and conversation evidence within the result's detail page. Extraction, toggle, and rejudge act on the **whole result row**, not an individual nested candidate or conversation node. Rejudging updates the representative result's verdict; recorded historical verdicts remain evidence from the original attack. Extraction preserves nested evidence but does not turn its items into standalone dataset entries. An extracted subset cannot establish the original run's full workload or success rate; preserve the source file and selection criteria.

### Reading older expanded result files

Earlier versions could write separate `<id>-attack-1`, `<id>-attack-2`, and subsequent rows. New runs no longer generate this format, but results analysis and the viewer continue to read it, including files mixed with normal single-row results.

- **Identity:** prefer `attack_parent_id` and `attack_parent_long_id`, with legacy suffix fallback. `attack_attempt` is an ordinal and `attack_invocation` identifies an outer invocation. Source-file provenance distinguishes otherwise identical IDs across runs.
- **Counts:** sum each saved row's additive `attempts`, never its ordinal. Group the original row and its attack rows as one dataset entry. A group succeeds if any corresponding row succeeds. Do not add a duplicate summary row.
- **Unjudged rows:** `success=null` is unjudged, excluded from failure extraction, and labelled UNJUDGED in the viewer. Overlapping per-turn conversation snapshots are not independent conversations.
- **Operations:** detail, toggle, rejudge, and extraction still target the selected unique row. Extraction renumbers rows while preserving `original_id`, `result_parent_id`, and `result_origin`, plus existing attack metadata. Dataset comparison associates attack outcomes with their source entry and counts success once per result file.

Inspect `attempt_history` and `conversation` before claiming that intermediate evidence is missing. Content absent from both cannot be reconstructed from an attempt count.
