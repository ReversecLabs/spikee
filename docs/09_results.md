# Spikee Results

The `spikee results` command includes several tools to aid in the analysis of test results.
* `spikee results analyze`: Processes results files to provide a detailed breakdown of performance statistics.
* `spikee results rejudge`: See [re-judging](<./07_judges.md#Re-judging>), revaluate results using a different judge or options.
* `spikee results extract`: Extracts user-defined categories of results from results files for further analysis.
* `spikee results dataset-comparison`: Compares the results of a single dataset across multiple targets to identify trends or differences in performance.
* `spikee results web-viewer`: Processes results files and displays them in a Flask web application for interactive exploration.
* `spikee results convert-to-excel`: Converts results files into Excel format.


# The `analyze` Command
## Usage Examples
```bash
# Analyze a single results file, providing all standard metrics and breakdowns
spikee results analyze --result-file results/results_openai_api-gpt-4o-mini_*.jsonl
```

### Multi-file Options
Spikee allows for multiple results files to be provided in a single analysis run using the `--result-file` and `--result-folder` flags.
* Results files processed sequentially in the provided order (folders are scanned and included in alphabetical order), combining all results into a single analysis.
* The `--result-folder` flag will scan the specified folder for all JSONL files starting with `results_`.
* Multi-file is not compatible with single dataset flags (e.g., `--false-positive-checks`).
* By default, when multiple results files are included in an analysis, each results file will be processed and outputted separately. To combine all results into a single analysis, use the `--combine` flag.

```bash
# Analyze multiple results files together.
spikee results analyze --result-file results/results_openai-api_cybersec-2026-01-*.jsonl \
                       --result-file results/results_openai-api_simsonsun-high-quality-jailbreaks-*.jsonl
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
*   **Total Attempts:** The sum of all individual requests made to the target. This reflects the total workload or API cost of the test, including all retries and dynamic attack iterations.
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
This command is used to extract specific categories of results from one or more results files for further analysis. It allows you to filter results based on success, failure, errors, guardrail triggers, or custom search queries.

## Usage Examples
```bash
# Extract all successful attacks to a new dataset file for further analysis
spikee results extract --result-file results/results_openai-api_cybersec-2026-01-*.jsonl \
                       --result-file results/results_openai-api_simsonsun-high-quality-jailbreaks-*.jsonl
                       --category success
```

```bash
# Extract custom search query for further analysis
spikee results extract --result-file results/results_openai-api_cybersec-2026-01-*.jsonl \
                       --result-file results/results_openai-api_simsonsun-high-quality-jailbreaks-*.jsonl \
                       --category custom \
                       --custom-search "error:Guardrail was triggered by the target"
```

## Extraction Categories
The `extract` command supports several categories for filtering results:
*   **success:** Extracts all successful entries.
*   **failure:** Extracts all failed entries.
*   **errors:** Extracts all entries containing errors.
*   **guardrail:** Extracts all entries where a guardrail was triggered.
*   **no-guardrail:** Extracts all entries where a guardrail was not triggered.
*   **custom:** Extracts entries matching a user-defined search query. 

### Custom Search Query
When using the `custom` category, you can specify a search query using the `--custom-search` flag.

Valid `--custom-search` queries include:
*  `search_term`: Searches all entry fields for the specified search_term.
*  `field:search_term`: Searches a specific field within an entry for the search_term.
*  `!search_term` or `!field:search_term`: Inverse search

Multiple `--custom-search` flags can be provided to add multiple search conditions. Entries must match *all* provided conditions to be included in the output.

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

# The `web-viewer` Command
This command processes results files and displays them in a Flask web application for interactive exploration.

## Set-Up
```bash
# Install additional requirements
pip install -r requirements-viewer.txt

# Add viewer components to workspace
spikee init --include-viewer

```

## Usage Examples
```bash
spikee results web-viewer --result-file results/results_cybersec-2025-04-*.jsonl 

# Specify a different host and port (Defaults, host=127.0.0.1, port=8080)
spikee results web-viewer --result-file results/results_cybersec-2025-04-*.jsonl --host 192.168.1.10 --port 8081
```