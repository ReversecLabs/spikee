**# Spikee Documentation - Table of Contents**
  
This page provides an index of all detailed guides for Spikee.

1. **[Cheatsheet](./01_cheatsheet.md)**
    *   A reference for all Spikee CLI commands, flags, and arguments.
  
2.  **[Built-in Seeds and Modules](./02_builtin.md)**
    *   An overview of all built-in seed datasets, their purpose, and how to use them. Includes built-in targets, plugins, attacks, and judges.
  
3.  **[Built-in Targets, Attacks and Judges](./03_builtin_targets_attacks_and_judges.md)**
    *   An overview of all built-in targets, attacks and judges.
  
4.  **[LLM Providers](./03_llm_providers.md)**
    *   An overview of the built-in LLM service, and a list of supported providers and models. Includes Billing Tracker.
  
5.  **[Dataset Generation](./04_dataset_generation.md)**
    *   A reference for `spikee generate`, explaining how to generate datasets and utilise arguments to modify datasets.
  
6.  **[Testing](./05_testing.md)**
    *   A reference for `spikee test`, explaining how to test an LLM, application or guardrail and utilise arguments to modify testing behavior.
  
7.  **[Creating Custom Targets](./06_custom_targets.md)**
    *   A guide to implementing target scripts for interacting with any LLM, application or guardrail. Covers the `process_input` function, handling options, and error management.
  
8.  **[Creating Custom Plugins](./07_custom_plugins.md)**
    *   Explains how to create plugins for transforming payloads during dataset generation. Covers the `transform` function and the difference between plugins and dynamic attacks.
    
9.  **[Creating Dynamic Attack Scripts](./08_dynamic_attacks.md)**
    *   Details how to build iterative attack scripts that apply real-time transformations. Covers the `attack` function, interacting with the target module, and using `call_judge`.
  
10. **[Judges: Evaluating Attack Success](./09_judges.md)**
    *   An explanation of the judge system for evaluating test outcomes. Covers basic judges, LLM-based judges, cloud vs. local models, and creating custom logic.
  
11. **[Testing Guardrails Using Spikee](./10_guardrail_testing.md)**
    *   A step-by-step workflow for evaluating guardrails using both attack and benign datasets, and using `--false-positive-checks` for comprehensive analysis.
  
12. **[Spikee Results](./11_results.md)**
    *   A guide to understanding the results analysis tools available in Spikee including `analyze` and `extract`. Explains core metrics, multi-file analysis, and various result commands.
  
13. **[Installing Spikee in an Isolated Environment](./12_installing_spikee_in_isolated_environments.md)**
    * A step-by-step guide on how to install Spikee in a test environment that has limited / no internet access.
  
14. **[Generating Custom Datasets with an LLM](./13_llm_dataset_generation.md)**
    *   Methods for using LLMs to generate use-case specific datasets. Covers creating `standalone_user_inputs.jsonl` and custom `instructions.jsonl` files.
  
15. **[Functional Testing Guide](./14_functional_testing.md)**
    *   Run the end-to-end CLI regression suite locally using pytest.