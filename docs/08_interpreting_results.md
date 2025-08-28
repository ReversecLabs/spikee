# Interpreting Spikee Results

The `spikee results analyze` command processes the output of a test run to provide performance statistics. This guide explains how to interpret these metrics to evaluate a target's security posture.

## Core Metrics

These top-level statistics provide a high-level summary of the test run.

*   **Total Unique Entries:** The number of distinct test cases processed. Each original prompt from your dataset is considered one unique entry, even if it triggers a dynamic attack with multiple iterations. This is the denominator for the overall success rate.
*   **Successful Attacks (Total):** The number of unique entries where at least one attempt (either a standard prompt or a dynamic attack iteration) was deemed successful by its Judge.
*   **Failed Attacks:** The number of unique entries where *all* attempts failed and no errors occurred.
*   **Errors:** The number of unique entries where *all* attempts resulted in an error (e.g., an API timeout or connection failure). An entry is only counted as an error if none of its attempts succeeded or failed normally.
*   **Total Attempts:** The sum of all individual requests made to the target. This reflects the total workload or API cost of the test, including all retries and dynamic attack iterations.
*   **Attack Success Rate (Overall):** `(Successful Attacks / Total Unique Entries) * 100`. This is the main metric for assessing the target's overall vulnerability during the test.

## Dynamic Attack Metrics

These metrics appear only if a dynamic attack was run using the `--attack` flag. They help you distinguish between vulnerabilities found by your initial dataset and those found by an adaptive attack script.

*   **Initially Successful:** The number of entries that succeeded using the standard prompt, *before* any dynamic attack was needed. This measures the baseline vulnerability to the static payloads in your dataset.
*   **Only Successful with Dynamic Attack:** The number of entries that succeeded *only* because the dynamic attack script found a working variation. This quantifies the additional risk uncovered by the attack strategy.
*   **Attack Success Rate (Improvement from Dynamic Attack):** The percentage of total entries that were only broken by the dynamic attack. This directly measures the value added by the adaptive attack logic.
*   **Dynamic Attack Statistics Table:** This table breaks down the performance of each dynamic attack script used:
    *   `Attack Type`: The name of the attack script.
    *   `Total`: How many times this dynamic attack script was *invoked* (i.e., for how many entries).
    *   `Successes`: How many times this script ultimately achieved a success.
    *   `Success Rate`: `(Successes / Total) * 100`. This shows how effective a specific attack strategy was *when it was used*.

## False Positive Analysis Metrics (For Guardrails)

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

## Breakdown and Combination Tables

These tables help you drill down into the results to identify specific patterns of vulnerability. They break down success rates by factors like `Jailbreak Type`, `Instruction Type`, `Plugin`, and `Language`.

By sorting these tables (they are sorted by success rate by default), you can quickly answer questions like:
*   Which type of jailbreak is most effective against my target?
*   Is my target more vulnerable to XSS-related instructions or data exfiltration?
*   Does Base64 encoding (via a plugin) increase or decrease the success rate?

## Actionable Insights and Limitations

The results from `spikee` can help you answer key security questions, but it's important to understand their context.

**Questions You Can Answer:**
*   **Overall Resilience:** What is the baseline success rate of prompt injection attacks against my application using a specific dataset?
*   **Specific Vulnerabilities:** Is my application susceptible to a specific class of attack, like data exfiltration using Markdown, as represented in the test set?
*   **Guardrail Efficacy:** How effective is my guardrail at blocking known attacks (Recall), and how much does it interfere with legitimate use (Precision)?
*   **Regression Testing:** Did a recent change to my application's prompt or model introduce a new vulnerability that was previously caught?

**Limitations and Caveats:**
*   **Dataset-Dependent Results:** The results are entirely dependent on the dataset used. A 0% success rate means the target was resilient to the *specific attacks in your dataset*, not that it is immune to all possible attacks. The space of potential jailbreaks and prompt injections is vast, and it is not feasible to create a dataset that covers all possibilities.
*   **Severity Not Measured:** The success rate does not measure the *impact* or *severity* of a successful attack. A successful "data exfiltration" attack is likely more severe than a successful "make a joke" attack, but both count as one success in the statistics. Manual review of successful attacks in the `results.jsonl` file is essential to understand the true risk.
