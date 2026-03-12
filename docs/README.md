**# Spikee Documentation - Table of Contents**
  
This page provides an index of all detailed guides for Spikee.
  
1.  **[Built-in Seeds, Datasets and Plugins](./01_builtin_datasets_and_plugins.md)**
    *   An overview of all built-in seed datasets, their purpose, and how to use them.
  
2.  **[Dataset Generation](./02_dataset_generation.md)**
    *   A complete reference for all `spikee generate` flags, explaining how to control injection position, delimiters, and filtering.
  
3.  **[Built-in Targets, Attacks and Judges](./03_builtin_targets_attacks_and_judges.md)**
    *   An overview of all built-in targets, attacks and judges.
  
4.  **[LLM Providers](./04_llm_providers.md)**
    *   List of supported provider prefixes, their required environment variables, and configuration examples for targets, attacks, and judges.
  
5.  **[Creating Custom Targets](./05_custom_targets.md)**
    *   A guide to implementing target scripts for interacting with any LLM, API, or application. Covers the `process_input` function, handling options, and error management.
  
6.  **[Creating Custom Plugins](./06_custom_plugins.md)**
    *   Explains how to create plugins for transforming payloads during dataset generation. Covers the `transform` function and handling `exclude_patterns`.
  
7.  **[Creating Dynamic Attack Scripts](./07_dynamic_attacks.md)**
    *   Details how to build iterative attack scripts. Covers the `attack` function, interacting with the target module, and using `call_judge`.
  
8.  **[Judges: Evaluating Attack Success](./08_judges.md)**
    *   An explanation of the judge system for evaluating test outcomes. Covers built-in judges, LLM-based judges, and creating custom logic.
  
9.  **[Testing Guardrails Using Spikee](./09_guardrail_testing.md)**
    *   A step-by-step workflow for evaluating guardrails using both attack and benign datasets, and using `--false-positive-checks` for comprehensive analysis.
  
10. **[Spikee Results](./10_results.md)**
    *   A guide to understanding the output of `spikee results analyze`. Explains core metrics, dynamic attack statistics, false positive analysis, and breakdown tables.
  
11. **[Generating Custom Datasets with an LLM](./11_llm_dataset_generation.md)**
    *   Methods for using LLMs to generate use-case specific datasets. Covers creating `standalone_user_inputs.jsonl` and custom `instructions.jsonl` files.
  
12. **[Installing Spikee in an Isolated Environment](./12_installing_spikee_in_isolated_environments.md)**
    *   A step-by-step guide on how to install spikee in a test environment that has limited / no internet access.
  
13. **[Functional Testing Guide](./13_functional_testing.md)**
    *   Run the end-to-end CLI regression suite locally.