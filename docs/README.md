# Spikee Documentation - Table of Contents

This page provides an index of all detailed guides for Spikee.

1.  **[Built-in Seeds and Datasets](./01_builtin_seeds_and_datasets.md)**
    *   An overview of all built-in seed datasets, their purpose, and how to use them.

2.  **[Dataset Generation Options](./02_dataset_generation_options.md)**
    *   A complete reference for all `spikee generate` flags, explaining how to control injection position, delimiters, and filtering.

3.  **[Creating Custom Targets](./03_custom_targets.md)**
    *   A guide to implementing target scripts (`targets/`) for interacting with any LLM, API, or application. Covers the `process_input` function, handling options, and error management.

4.  **[Developing Custom Plugins](./04_custom_plugins.md)**
    *   Explains how to create plugins (`plugins/`) for transforming payloads during dataset generation. Covers the `transform` function and handling `exclude_patterns`.

5.  **[Writing Dynamic Attack Scripts](./05_dynamic_attacks.md)**
    *   Details how to build iterative attack scripts (`attacks/`). Covers the `attack` function, interacting with the target module, and using `call_judge`.

6.  **[Judges: Evaluating Attack Success](./06_judges.md)**
    *   An explanation of the judge system for evaluating test outcomes. Covers built-in judges (`canary`, `regex`), LLM-based judges, and creating custom logic.

7.  **[Testing Guardrails](./07_guardrail_testing.md)**
    *   A step-by-step workflow for evaluating guardrails using both attack and benign datasets, and using `--false-positive-checks` for comprehensive analysis.

8.  **[Interpreting Spikee Results](./08_interpreting_results.md)**
    *   A guide to understanding the output of `spikee results analyze`. Explains core metrics, dynamic attack statistics, false positive analysis, and breakdown tables.

9.  **[Generating Custom Datasets with an LLM](./09_llm_dataset_generation.md)**
    *   Methods for using LLMs to generate use-case specific datasets. Covers creating `standalone_user_inputs.jsonl` and custom `instructions.jsonl` files.

10. **[Adaptive Random Suffix Attack](./10_adaptive_rsa.md)**
    *   A comprehensive guide to the Adaptive RSA attack, including configuration options, dual optimization features, and guardrail testing workflows.