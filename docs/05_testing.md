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

Result filenames use compact labels: target options that need cleaning or shortening
become up to 24 readable characters plus a 12-character hash of the full options.
Long dataset names and tags are also shortened, keeping generated filenames under
180 characters. The original options are still passed to the target unchanged.
Automatic resume recognizes both compact names and existing result filenames.

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
Resuming appends new results to the selected file, including with `--auto-resume` or the interactive picker. Completed entries are skipped; repeated resumes do not create additional results files. An unfinished entry is retried from the start, with any older incomplete history rows replaced. Use `--no-auto-resume` to start a fresh file.

By default, Spikee will analyse the workspace results folder to identify an previously executed test files with a matching name pattern. It will then display an interactive prompt to start a new test or to continue a previous test. The following flags can be used to control this behavior:
- `--resume-file` - append to an existing results JSONL file. Only works with a single dataset. A missing or malformed file raises an error.
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
## Attack history and conversation logs

Dynamic attacks write one result per dataset entry, with ID `<id>-attack`, the successful or final failed input/response, and the overall `attempts` count. Supporting single-turn attacks, including `best_of_n`, `llm_jailbreaker`, and `llm_multi_language_jailbreaker`, also include an `attempt_history` list containing the intermediate inputs, responses, and their recorded verdicts. History is enabled by default; no CLI option or viewer checkbox is required.

To omit this optional history, set `SPIKEE_ATTACK_HISTORY=false` in the environment or workspace `.env`:

```bash
SPIKEE_ATTACK_HISTORY=false spikee test \
    --dataset datasets/my-dataset.jsonl --target my_target \
    --attack best_of_n --attack-iterations 20
```

`false`, `0`, `no`, and `off` disable recording (case-insensitive). The setting changes memory and result-file usage, without changing target calls, judge calls, stopping conditions, the representative result, or attempt counts. Custom and legacy modules may omit history; Spikee continues to accept their existing return tuple and does not reconstruct discarded candidates.

**Crescendo, GOAT, Echo Chamber, and `multi_turn` keep their existing `conversation` logs.** They do not use `SPIKEE_ATTACK_HISTORY` or duplicate their turns in `attempt_history`. The graph-based attacks retain recorded turns and backtracking branches in one row; inspect the conversation in the viewer before concluding that earlier exchanges were lost. Existing error paths can omit some details.

With repeated invocations using `--attempts`, provided candidate histories are combined in the one attack result. Each nested item receives a 1-based `invocation` number. The top-level input, response, and conversation remain from the successful or final invocation. Only its conversation graph survives; candidate history does not change conversation logging.

Twenty dynamic iterations still contribute 20 to total attempts; a separate standard attempt adds its own count. Nested history items and conversation nodes are evidence, not additional dataset entries or counts. See [Dynamic Attacks](08_dynamic_attacks.md#attempt-history) and [Results Analysis](11_results.md#attack-history-and-conversation-results).

Resume preserves completed results as saved. Start a fresh run with `--no-auto-resume` to collect history for previously completed entries that lack it. History is held in memory until the attack returns; it is not a checkpoint for resuming within an attack.
