# Built-in Seed Datasets

Spikee comes with a variety of built-in seed datasets, each designed for a specific testing purpose. These seeds are located in the `datasets/` directory after you run `spikee init`. You can list them at any time with `spikee list seeds`.

This guide provides an overview of each available seed folder.

---

### `seeds-cybersec-2026-01`
*   **Purpose:** A general-purpose dataset for testing prompt injection. It focuses on common attack goals seen in web application security, such as data exfiltration, cross-site scripting (XSS), and resource exhaustion.

### `seeds-in-the-wild-jailbreak-prompts`
*   **Purpose:** Contains approximately 1,400 real-world jailbreak prompts collected from public sources like Discord and Reddit (filtered from the TrustAIRLab dataset). Ideal for testing a target's resilience against known, publicly available jailbreaks.
*   **Note:** This dataset requires you to run a fetch script to download the prompts. See the `README.md` inside the seed folder for instructions. It uses an LLM judge by default.

### `seeds-simsonsun-high-quality-jailbreaks`
*   **Purpose:** A high-quality set of contamination-free jailbreak prompts, specifically curated to avoid overlap with the training data of many common safety classifiers.
*   **Note:** This dataset requires you to run a fetch script. See the `README.md` inside the seed folder.

### `seeds-wildguardmix-harmful`
*   **Purpose:** A dataset for testing harmful content generation. The prompts are sourced from the WildGuard-Mix dataset.
*   **Note:** This dataset requires you to run a fetch script. See the `README.md` inside the seed folder. It uses an LLM judge by default.

### `seeds-wildguardmix-harmful-fp`
*   **Purpose:** A companion dataset to `seeds-wildguardmix-harmful`, containing benign (harmless) prompts. This dataset is intended for use with the `--false-positive-checks` flag to measure how often a guardrail incorrectly blocks legitimate prompts when evaluating harmful content filters.

### `seeds-toxic-chat`
*   **Purpose:** A dataset for testing toxic prompts, filtered from 10K user prompts collected from the Vicuna online demo.
*   **Note:** This dataset requires you to run a fetch script. See the `README.md` inside the seed folder. It uses an LLM judge by default.

### `seeds-investment-advice`
*   **Purpose:** Designed to test topical guardrails that are supposed to block personal financial or investment advice. It includes both malicious instructions and standalone attack prompts.

### `seeds-investment-advice-fp`
*   **Purpose:** A companion dataset to `seeds-investment-advice`, containing benign (harmless) queries about financial topics. This dataset is intended for use with the `--false-positive-checks` flag to measure how often a guardrail incorrectly blocks legitimate prompts.

### `seeds-sysmsg-extraction-2025-04`
*   **Purpose:** Specifically designed to test for system prompt extraction. The instructions and judges are tailored to detect if the target model leaks its own system prompt or initial instructions.

### `seeds-llm-mailbox`
*   **Purpose:** An example seed tailored for testing an email summarization feature. The documents are sample emails, and the instructions are designed to test for vulnerabilities in that specific context. See the associated [blog post](https://labs.reversec.com/posts/2025/01/spikee-testing-llm-applications-for-prompt-injection) for a detailed walkthrough.

---

### Utility Seeds

### `seeds-mini-test`
*   **Purpose:** A very small set of examples for quick, functional testing of Spikee itself. Use this to verify your setup or to test a new custom target or plugin without running a large number of tests.

### `seeds-empty`
*   **Purpose:** An empty template folder. It contains empty `documents.jsonl`, `jailbreaks.jsonl`, and `instructions.jsonl` files. This is the recommended starting point when creating a new dataset from scratch, especially for standalone attacks.

### `seeds-harmful-instructions-only`
*   **Purpose:** Specifically designed for **LLM-driven attacks** such as **Crescendo** and **LLM-Jailbreaker**. These attacks require only a plain instruction (objective) to generate their own attack vectors dynamically.
*   **Content:** Contains harmful instructions in `instructions.jsonl`, while leaving jailbreaks and user inputs as empty placeholders.

### `zzz_LEGACY_seeds-targeted-2024-12`
*   **Purpose:** A legacy dataset that has been superseded by `seeds-cybersec-2026-01`. It is kept for backward compatibility.

# Build-in Plugins

Spikee includes several build-in plugins, that can be leverages to enhance dataset generation. These are scripts that will apply static transformations to payloads during dataset generation, creating multiple iterations of each entry. Built-in plugins are located in the `spikee/plugins/` directory, local plugins are located in the `plugins/` directory within your workspace. You can list them at any time with `spikee list plugins`.

You can customize the behavior of attacks using the following command-line options:
* `--plugin-options`: Passes a single string option to the plugin script for custom behavior (e.g., `"<plugin_name>:variants=50"`).

The following list provides an overview of each build-in plugin, further information on each plugin can be found within the plugin file.


## Basic Plugins
* `1337`: Transforms text into "leet speak" by replacing certain letters with numbers or symbols.
* `ascii_smuggler`: Transforms ASCII text into a series of Unicode rags that are generally invisible to most UI elements (bypassing content filters).
* `base64`: Encodes text using Base64 encoding.
* `ceasar`: Applies a Caesar cipher to the text, shifting letters by a specified number of positions.
    * Options: `shift` (number of positions to shift, default: 3).
* `hex`: Encodes text into its hexadecimal representation.
* `morse`: Encodes text into Morse code.
* `splat`: Obfuscates the text using splat-based techniques (e.g., asterisks '*', special characters,
and spacing tricks), to bypass basic filters.


## Attack Plugins

These plugins are based on dynamic attack techniques, but have been adapted to work as static transformations during dataset generation.

* `anti_spotlighting`: Generates variations of delimiter-based attacks to test
LLM applications against spotlighting vulnerabilities.
    * Options: `variants` (number of variations to generate, default: 50).
* `best_of_n`: Implements ["Best-of-N Jailbreaking" John Hughes et al., 2024](https://arxiv.org/html/2412.03556v1#A1) to apply character scrambling, random capitalization, and character noising.
    * Options: `variants` (number of variations to generate, default: 50).
* `prompt_decomposition`: Decomposes a prompt into chunks and generates shuffled variations.
    * Options: 
        `modes` (LLM model to apply, default: dumb),
        `variants` (number of variations to generate, default: 50).

## Usage Example
```bash
# base64 and best_of_n plugin
spikee generate --seed-folder datasets/seeds-cybersec-2026-01 \
                --plugin base64 best_of_n \
                --plugin-options "best_of_n:variants=20"
```