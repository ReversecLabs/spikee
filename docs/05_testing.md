# Testing Targets

The `spikee test` command is used to evaluate the security of an LLM, application or guardrail against a dataset of entries. The command offers several flags and arguments to control how entries are executed, and to configure the target, judge and attack modules used for testing.

Jump to Links:
- [Testing a Target](#testing-a-target)
- [Dataset Selection](#dataset-selection)
- [Targets](#targets)
- [Judges](#judges)
- [Attacks](#attacks)
- [Multi-Processing and Attempts Configuration](#multi-processing-and-attempts-configuration)
- [Sampling](#sampling)

## Testing a Target
Once you've generated a dataset, you can test it against a target using the `spikee test` command. The most basic usage:

```bash
spikee test 
    --dataset <path_to_dataset> \
    --target <target_name>
```

### Dataset Selection
- `--dataset` - specify a single dataset file (e.g. `./datasets/my_dataset.jsonl`)
- `--dataset-folder` - specify a folder containing multiple JSONL dataset files (e.g. `./datasets/`)

These args can be used multiple times to specify multiple datasets, which `spikee test` will assess sequentially against the target. However, at least one of `--dataset` or `--dataset-folder` is **required** to run a test.

```bash
spikee test 
    --dataset ./dataset/cybersec-2026-01.jsonl \
    --dataset ./dataset/simsonsun-high-quality-jailbreaks.jsonl \
    --dataset-folder ./dataset/cyber_datasets/ \
    --target llm_provider \
    --target-options "bedrock/claude45-haiku"
```

### Targets
- `--target` - specify a single target, using the filename (e.g., `./targets/example_chatbot.py` -> `example_chatbot`)
- `--target-options` - specify target defined options

Run `spikee list targets` to validate the target is in a loadable location, and to view supported options. A list of built-in targets is also available in [Built-In Targets](./02_builtin.md#built-in-targets).

```bash
spikee test 
    --dataset ./dataset/cybersec-2026-01.jsonl \
    --target llm_provider \
    --target-options "bedrock/claude45-haiku"
```

### Judges
- `--judge-options` - specify judge defined options. LLM judges typically require a `model=<model>` option to specify the LLM agent's model.

Run `spikee list judges` to view supported options. A list of built-in judges is also available in [Built-In Judges](./02_builtin.md#built-in-judges), and a list of supported LLM providers and models for LLM judges is available in [LLM Providers](./03_llm_providers.md#llm-provider-prefixes).

```bash
spikee test 
    --dataset ./dataset/cybersec-2026-01.jsonl \
    --target llm_provider \
    --target-options "bedrock/claude45-sonnet" \
    --judge-options "model=bedrock/claude45-haiku"
```

### Attacks
- `--attack` - specify an attack module
- `--attack-iterations` - specify the number of attack iterations / turns to perform for each dataset entry
- `--attack-options` - specify attack defined options
- `--attack-only` - skip the core testing loop and only run the attack module for each dataset entry.

By default, `spikee test` will initially execute a dataset entry against the target, and then only run the attack module if the core testing loop fails. The `--attack-only` flag skips the core testing loop and immediately runs the attack module for each dataset entry, which is useful for testing instructional or objective-based attacks that don't require an initial target execution.

Run `spikee list attacks` to view supported options. A list of built-in attacks is also available in [Built-In Attacks](./02_builtin.md#built-in-attacks).

```bash
spikee test 
    --dataset ./dataset/cybersec-2026-01.jsonl \
    --target llm_provider \
    --target-options "bedrock/claude45-sonnet" \
    --attack goat \
    --attack-options 'model=bedrock/deepseek-v3' \
    --attack-iterations 10 \
    --attack-only
```

### Multi-Processing and Attempts Configuration
- `--threads` - specify the number of threads for parallel processing (default: 4)
- `--attempts` - specify the number of attempts per entry (default: 1)
- `--max-retries` - specify the number of retries for rate-limiting controls / 429 errors (default: 3)
- `--throttle` - specify the time in seconds to wait between entries per thread (default: 0)

### Sampling
- `--sample` - specify a percentage of a dataset to sample (e.g., `0.15` for 15% of the dataset)
- `--sample-seed` - specify a seed for random sampling (default: 42)

```bash
# This will test 10% of randomly sampled entries from the dataset
spikee test 
    --dataset ./dataset/cybersec-2026-01.jsonl \
    --target llm_provider \
    --target-options "bedrock/claude45-sonnet" \
    --sample 0.1 \ 
    --sample-seed 123
```

### Resume
By default, Spikee will analyse the workspace results folder to identify an previously executed test files with a matching name pattern. It will then display an interactive prompt to start a new test or to continue a previous test. The following flags can be used to control this behavior:
- `--resume-file` - specify a path to a results JSONL file to resume from. Only works with a single dataset.
- `--auto-resume` - silently pick the latest matching results file if present
- `--no-auto-resume` - create new results file, do not attempt to resume

```bash
# This will resume from a specific results file
spikee test
    --dataset ./dataset/cybersec-2026-01.jsonl \
    --target llm_provider \
    --target-options "bedrock/claude45-sonnet" \
    --resume-file ./results/cybersec_test_results_2024-07-01.jsonl

# This will attempt to automatically resume from the latest results file for this test, if it exists
spikee test 
    --dataset ./dataset/cybersec-2026-01.jsonl \
    --target llm_provider \
    --target-options "bedrock/claude45-sonnet" \
    --auto-resume

# This will create a new results file, even if a previous results file exists
spikee test
    --dataset ./dataset/cybersec-2026-01.jsonl \
    --target llm_provider \
    --target-options "bedrock/claude45-sonnet" \
    --no-auto-resume

```