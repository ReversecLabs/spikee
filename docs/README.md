# Spikee Documentation

## Guides
- **[Quick Start Guide](../README.md#a-quick-start-guide-to-using-spikee)** - A concise walkthrough of the core Spikee workflow, from installation to results analysis.
- **[How to Spikee](./how-to-spikee/README.md)** - A step-by-step tutorial, covering using Spikee for testing, and developing custom modules.

## Reference Pages
- **[Cheatsheet](./01_cheatsheet.md)** - A reference for all Spikee CLI commands, flags, and arguments.
- **[Built-in Seeds and Modules](./02_builtin.md)** - A reference for all built-in seeds, targets, plugins, attacks, and judges.
- **[LLM Providers](./03_llm_providers.md)** - An overview of the built-in LLM utility, and provider modules.
  
## Generation and Testing Workflows
- **[Dataset Generation](./04_dataset_generation.md)** - A reference for `spikee generate`, explaining how to generate datasets and utilise arguments to modify datasets.
  
- **[Testing](./05_testing.md)** - A reference for `spikee test`, explaining how to test an LLM, application or guardrail and utilise arguments to modify testing behavior.

- **[Testing Guardrails Using Spikee](./10_guardrail_testing.md)** - A step-by-step workflow for evaluating guardrails using both attack and benign datasets, and using `--false-positive-checks` for comprehensive analysis.

- **[Spikee Results](./11_results.md)** - A guide to understanding the results analysis tools available in Spikee including `analyze` and `extract`. Explains core metrics, multi-file analysis, and various result commands.
  
## Custom Modules and Development
- **[Creating Custom Targets](./06_custom_targets.md)** - A guide to implementing target modules for interacting with any LLM, application or guardrail.
  
- **[Creating Custom Plugins](./07_custom_plugins.md)** - A guide to implementing plugin modules for transforming payloads during dataset generation.
    
- **[Creating Dynamic Attack Scripts](./08_dynamic_attacks.md)** - A guide to implementing dynamic attack modules, for real-time generation of attack payloads.
  
- **[Judges](./09_judges.md)** - An introduction to judge modules, used to evaluate target responses. Covers basic and LLM judges, their implementation and usage.

- **[Creating Provider Modules](./03_llm_providers.md#creating-custom-provider-modules)** - A guide to implementing provider modules for integrating any LLM with Spikee's.
  

## Additional Resources
- **[Installing Spikee in an Isolated Environment](./12_installing_spikee_in_isolated_environments.md)** - A step-by-step guide on how to install Spikee in a test environment that has limited / no internet access.
  
- **[Generating Custom Datasets with an LLM](./13_llm_dataset_generation.md)** - Methods for using LLMs to generate use-case specific datasets. Covers creating `standalone_user_inputs.jsonl` and custom `instructions.jsonl` files.
  
- **[Functional Testing Guide](./14_functional_testing.md)** - Run the end-to-end CLI regression suite locally using pytest.
