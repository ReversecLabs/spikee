# Spikee CLI Cheatsheet

This is a quick CLI reference guide for using Spikee, containing commands, arguments and links to more detailed documentation.

Jump to Links:
- [`spikee init`](#spikee-init)
- [`spikee list`](#spikee-list)
- [`spikee generate`](#spikee-generate)
- [`spikee test`](#spikee-test)
- [`spikee results`](#spikee-results)
- [`spikee viewer`](#spikee-viewer)

## `spikee init`
Initializes a new Spikee project in the current directory.

```bash
mkdir workspace && cd workspace
spikee init
```

| Argument            | Description                                            | Docs     |
| ------------------- | ------------------------------------------------------ | -------- |
| `--force`           | Overwrite existing directories                         |          |
| `--include-builtin` | Copy built-in modules to local workspace               |          |
| `--include-viewer`  | Include the built-in web viewer in the local workspace | [Link](./11_results.md#the-web-viewer-command) |

## `spikee list`
Lists all local and built-in modules available. Use `--description` to include module descriptions.

```bash
spikee list seeds
spikee list datasets
spikee list targets
spikee list plugins
spikee list attacks
spikee list judges
spikee list providers --description
```

## `spikee generate`
Generates a dataset from a seed folder.

```bash
spikee generate --seed-folder my_seeds
```
| **Source Arguments**          | Description                                        | Docs     |
| ----------------------------- | -------------------------------------------------- | -------- |
| `--seed-folder`               | **Required** Path of seed folder                   | |
| `--include-standalone-inputs` | Include standalone_user_inputs.jsonl               | [Link](./04_dataset_generation.md#standalone-prompts---include-standalone-inputs) |
| `--include-system-message`    | Includes system_messages.toml                      | [Link](./04_dataset_generation.md#modifying-prompts-with-system-messages-prefixes-and-suffixes) |
| `--tag`                       | Include an identifying tag in the dataset filename |          |


| **Transformation Arguments** | Description                                                                                | Docs     |
| ---------------------------- | ------------------------------------------------------------------------------------------ | -------- |
| `--plugins`                  | Space-separated list of plugins                                                            | [Link](./04_dataset_generation.md#plugins) |
| `--plugin-options`           | Apply options as `"plugin1:option1=value1,option2=value2;plugin2:option2=value2"`          |          |
| `--plugin-only`              | Only output plugin entries                                                                 |          |
| `--include-fixes`            | Comma-separated list (e.g., 'adv_prefixes', 'adv_suffixes', prefixes=<filename>, suffixes=<filename>) | [Link](./04_dataset_generation.md#modifying-prompts-with-system-messages-prefixes-and-suffixes) |

| **Formatting Arguments**      | Description                                                                               | Docs     |
| ----------------------------- | ----------------------------------------------------------------------------------------- | -------- |
| `--format`                    | Output format: user-input (default / apps), full-prompt (LLMs), or burp                   |  |
| `--languages`                 | Comma-separated list of languages to filter jailbreaks and instructions (e.g., en)        | [Link](./04_dataset_generation.md#filtering-for-focused-datasets) |
| `--match-languages`           | Only combine jailbreaks and instructions with matching languages (default: True)          | [Link](./04_dataset_generation.md#filtering-for-focused-datasets) |
| `--positions`                 | Position to insert jailbreaks (start, middle, end). ignored if `<PLACEHOLDER>` is present | [Link](./04_dataset_generation.md#controlling-injection-position---positions) |
| `--injection-delimiters`      | Delimiters to use when injecting jailbreaks (default: `\nINJECTION_PAYLOAD\n`)            | [Link](./04_dataset_generation.md#controlling-the-injection-boundary---injection-delimiters) |
| `--spotlighting-data-markers` | Comma-separated list of data markers (placeholder: "DOCUMENT")                            | [Link](./04_dataset_generation.md#testing-a-defense-spotlighting-markers---spotlighting-data-markers) |
| `--instruction-filter`        | Comma-separated list of instruction types to include                                      | [Link](./04_dataset_generation.md#filtering-for-focused-datasets) |
| `--jailbreak-filter`          | Comma-separated list of jailbreak types to include                                        | [Link](./04_dataset_generation.md#filtering-for-focused-datasets) |

## `spikee test`
Tests a dataset against a target.

```bash
spikee test --dataset my_dataset.jsonl --target my_target
```

| **Dataset Arguments** | Description                                                | Docs     |
| --------------------- | ---------------------------------------------------------- | -------- |
| `--dataset`           | Path to a dataset file                                     | [Link](./05_testing.md#dataset-selection) |
| `--dataset-folder`    | Path to a dataset folder containing multiple dataset files | |

(NB, These args can be used multiple times to specify multiple datasets, at least one of `--dataset` or `--dataset-folder` is **required**)

| **Module Arguments** | Description                                                 | Docs     |
| -------------------- | ----------------------------------------------------------- | -------- |
| `--target`           | **Required** Name of the target to test (local or built-in) | [Link](./05_testing.md#targets) |
| `--target-options`   | Options to pass to the target                               |          |
| `--judge-options`    | Options to pass to the judge                                | [Link](./05_testing.md#judges) |

| **Testing Arguments** | Description                                               | Default | Docs     |
| --------------------- | --------------------------------------------------------- | ------- | -------- |
| `--threads`           | Number of threads for parallel processing                 | 4       | [Link](./05_testing.md#multi-processing-and-attempts-configuration) |
| `--attempts`          | Number of attempts per entry                              | 1       | [Link](./05_testing.md#multi-processing-and-attempts-configuration) |
| `--max-retries`       | Number of retries for rate-limiting controls / 429 errors | 3       | [Link](./05_testing.md#multi-processing-and-attempts-configuration) |
| `--throttle`          | Time in seconds to wait between entries per thread        | 0       | [Link](./05_testing.md#multi-processing-and-attempts-configuration) |
| `--sample`            | Only test a sample of a dataset (e.g., 0.15 for 15%)      | 1       | [Link](./05_testing.md#sampling) |
| `--sample-seed`       | Seed for random sampling                                  | 42      | [Link](./05_testing.md#sampling) |
| `--tag`               | Include a tag at the end of the results filename          |         |          |

| **Attack Arguments**  | Description                                                             | Docs     |
| --------------------- | ----------------------------------------------------------------------- | -------- |
| `--attack`            | Name of the attack module                                               | [Link](./05_testing.md#attacks) |
| `--attack-iterations` | Number of attack iterations / maximum number of turns per dataset entry | [Link](./05_testing.md#attacks) |
| `--attack-options`    | Options to pass to the attack module                                    | [Link](./05_testing.md#attacks) |
| `--attack-only`       | Only run the attack module without standard attempts                    | [Link](./05_testing.md#attacks) |

(NB, attacks are only run if the core test flow fails, or `--attack-only` is set)


| **Resume Arguments** | Description                                                                   | Docs     |
| -------------------- | ----------------------------------------------------------------------------- | -------- |
| `--resume-file`      | Path to a results JSONL file to resume from. Only works with a single dataset | [Link](./05_testing.md#resume) |
| `--auto-resume`      | Silently pick the latest matching results file if present                     | [Link](./05_testing.md#resume) |
| `--no-auto-resume`   | Create new results file, do not attempt to resume                             | [Link](./05_testing.md#resume) |

## `spikee results`

### `analyze`
Analyzes results files to produce aggregate statistics and visualizations - [Link](./11_results.md#the-analyze-command)

```bash
spikee results analyze --results-file results.jsonl
```
| Argument            | Description                                            |
| ------------------- | ------------------------------------------------------ |
| `--results-file`    | Path to a results JSONL file |
| `--results-folder`  | Path to a folder containing multiple results JSONL files |
| `--false-positive-checks` | Path to a JSONL file with benign prompts for false positive analysis. Only works with a single dataset. |
| `--output-format`   | Output format: console (default) or html                 |
| `--overview`        | Only output the general statistics of results files     |
| `--combine`         | Combine results from multiple files into a single analysis. |

(NB, Result file/folder args can be used multiple times to specify multiple datasets, but at least one is **required**)

### `rejudge`
Rejudges a results file - [Link](./09_judges.md#2-perform-rejudging)

```bash
spikee results rejudge --results-file results.jsonl
```
| Argument            | Description                                            |
| ------------------- | ------------------------------------------------------ |
| `--results-file`    | Path to a results JSONL file |
| `--results-folder`  | Path to a folder containing multiple results JSONL files |
| `--judge-options`   | Options to pass to the judge |
| `--resume`         | Attempt to resume the most recent re-judge file |

(NB, Result file/folder args can be used multiple times to specify multiple datasets, but at least one is **required**)

### `extract`
Searches for result entries by category or search term and extracts them to another JSONL file - [Link](./11_results.md#the-extract-command)

```bash
spikee results extract --results-file results.jsonl --category "success"
```

| Argument            | Description                                            |
| ------------------- | ------------------------------------------------------ |
| `--results-file`    | Path to a results JSONL file |
| `--results-folder`  | Path to a folder containing multiple results JSONL files |
| `--category`        | Extracts prompts by category: success (default), failure, error, guardrail, no-guardrail, custom |
| `--custom-search`   | Custom search string to filter prompts when --category=custom. Formats: 'search_string', 'field:search_string' or '!search_string' to invert match |
| `--tag`             | Include a tag at the end of the results filename          |

(NB, Result file/folder args can be used multiple times to specify multiple datasets, but at least one is **required**)

## `spikee viewer`
Launches the Spikee web viewer.

> Viewer is a WIP, currently only the results viewer is available.

### Common Arguments
| Argument            | Description                                            |
| ------------------- | ------------------------------------------------------ |
| `-h`, `--host`      | Host address for the web viewer (default: 127.0.0.1)   |
| `-p`, `--port`      | Port number for the web viewer (default: 8080)         |
| `-d`, `--debug`     | Enable debug mode for the web viewer (default: False)  |
| `--truncate`        | Truncate long fields in the web viewer for better readability (default: 500 characters, disable with 0) |

**Usage**
```bash
spikee viewer <common args> results <results-specific args>
```

### `results` Specific Arguments - [Link](./11_results.md#the-web-viewer-command)
| Argument            | Description                                            |
| ------------------- | ------------------------------------------------------ |
| `--result-file`      | Path to a results JSONL file, generated using the dataset |
| `--result-folder`    | Path to a results folder containing multiple JSONL files, generated using the dataset |
| `--allow-ast`        | Allow AST parsing in the web viewer (use with caution) |

(NB, Result file/folder args can be used multiple times to specify multiple datasets, but at least one is **required**)


```bash
spikee viewer -p 8081 results --result-folder .\results\
```

