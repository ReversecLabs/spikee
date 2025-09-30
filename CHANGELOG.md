# Changelog

All notable changes to this project will be documented in this file.

## [0.4.2] - 2025-09-27

### Added

- Spikee now prompts the user to auto-resume an interrupted test if it finds results files in the results folder. This behaviour can be controlled with `--auto-resume` and `no-auto-resume`. 

### Changed

- Fixed typos -> `llamaccp` -> `llamacpp`
- Python code is now autimatically linted/formatted by the GitHub release action.

## [0.4.1] - 2025-09-20

### Changed

We changed seed dataset names and generation flags to reflect how Spikee is actually used today.  
Over time, “documents” vs “inputs” and `--standalone-attacks` vs `--include-standalone-inputs` created confusion, and defaults like language matching and full prompts evolved. This release cleans that up so the CLI and seed files match current practice while still keeping backward compatibility.

* **Seed Dataset Standardization**
  - `base_documents.jsonl` → `base_user_inputs.jsonl`
  - `standalone_attacks.jsonl` → `standalone_user_inputs.jsonl`
  - READMEs and CLI examples updated accordingly.

* **CLI Flags Simplified**
  - `--format user-input` is now the canonical format.  
    Replaces `--format document`; the old value is mapped automatically with a warning.
  - `--include-standalone-inputs` is the new flag for standalone prompts.  
    Looks for `standalone_user_inputs.jsonl` (or falls back to legacy `standalone_attacks.jsonl`).  
    If `--standalone-attacks <path>` is used, Spikee ignores the path, enables `--include-standalone-inputs`, and prints a deprecation warning.
  - Language matching is now **enabled by default** (`--match-languages true`).  
    Use `--match-languages false` to disable cross-language filtering.

### Notes

All old filenames and flags still work with warnings.  
This release aligns Spikee’s seeds and CLI options with the workflows users actually rely on, while giving everyone time to adapt.


## [0.4.0] - 2025-09-18

### Added

* **Rejudge Feature**
  - Spikee now supports the possibility of rejudging a results file. This comes handy if you want to try a different LLM judge or if you are in a situation where you can't call an LLM judge, so you just collect the LLm responses and then you can judge the results at a later stage when you can connect to an LLM.

## [0.3.1] - 2025-07-01

### Added

* **Options Support for Plugins and Attacks:**
  - Plugins and attacks can now implement `get_available_option_values()` to provide configurable options.
  - Added `--plugin-options` flag to `spikee generate` for plugin-specific configuration using format `"plugin1:option1;plugin2:option2"`.
  - Added `--attack-options` flag to `spikee test` for attack-specific configuration (e.g., `"mode=gpt4o-mini"`).
  - Plugin `transform()` and attack `attack()` functions now accept optional configuration parameters.
  - Common option patterns: `variants=N` for variation count, `mode=X` for algorithm selection.
  - `spikee list plugins` and `spikee list attacks` now display available options with defaults highlighted.

* **Enhanced Built-in Modules:**
  - **Plugins:**
    - Updated `best_of_n` and `anti_spotlighting` plugins to support `variants=N` options (default: `variants=50`).
    - New unified `prompt_decomposition` plugin with mode selection and variant control.
  - **Attacks:**
    - New unified `prompt_decomposition` attack with mode selection.
    - Updated `sample_attack` to demonstrate attack options with strategy selection.
  - **Mode options for both:** `mode=dumb` (default), `mode=gpt4o-mini`, `mode=gpt4.1-mini`, `mode=ollama-*` (gemma3, llama3.2, mistral-nemo, phi4-mini).

* **Default Option Handling:**
  - Targets, judges, plugins, and attacks automatically use their first available option as default when no options are specified.
  - Updated filename generation to include default options in result filenames for clarity.
  - Backward compatibility maintained for all module types without option support.

### Improved

* **Option Discovery and Display:**
  - All `spikee list` commands now show available options with defaults highlighted.
  - Consistent "first option is default" pattern across targets, judges, plugins, and attacks.
  - Changed option format from `key-value` to `key=value` for clarity (e.g., `mode=gpt4o-mini`, `variants=50`).

### Examples

```bash
# Plugin options
spikee generate --plugins best_of_n --plugin-options "best_of_n:variants=100"
spikee generate --plugins anti_spotlighting prompt_decomposition --plugin-options "anti_spotlighting:variants=15;prompt_decomposition:variants=10,mode=gpt4o-mini"

# Attack options  
spikee test --attack prompt_decomposition --attack-iterations 10 --attack-options "mode=gpt4o-mini"
```

## [0.3.0] - 2025-06-19

### Added

* **Runtime Option Flags:**  
  - `--judge-options <model>` and `--target-options <model>` for all `spikee test` calls.

* **Local LLMs for Judges (Ollama):**  
  - Use local Ollama models for judging (e.g. in `wildmix-harmful`, `in-the-wild-jailbreak-prompts`, `simsonsun-high-quality-jailbreaks`).  
  - Specify via `--judge-options`, e.g.:  
    ```bash
    spikee test --dataset datasets.jsonl --target my_target --judge-options ollama-gemma3
    ```

* **Unified Target APIs & Runtime Option Support:**  
  - Collated all individual target scripts into a small set of `<provider>_api.py` modules (`openai_api.py`, `togetherai_api.py`, `google_api.py`, `deepseek_api.py`, `aws_bedrock_api.py`, `azure_api.py`, `groq_api.py`, `ollama_api.py`).  
  - Each now accepts a `--target-options` string to pick the exact model/deployment at runtime, reducing duplicated code.  
  - Available targets and their valid options are discoverable via:
    ```bash
    spikee list targets
    ```

### Improved

* **Investment-Advice Judge Prompt:**  
  - Improved juding prompt for the `investment-advice` dataset.

## [0.2.3] - 2025-06-16

### Added

* **Dataset Sampling for Testing:**
    * Added `--sample <percentage>` flag to `spikee test` to randomly sample a subset of the dataset (e.g., `--sample 0.15` for 15%).
    * Added `--sample-seed <value>` flag to control sampling reproducibility. Default: `42`. Use `"random"` for a random seed (printed to console).
    * Sampling works correctly with `--resume-file`, maintaining the same sample set when resuming interrupted tests.
    * Useful for testing with large datasets under time or API quota constraints.

* **New Seed Datasets:**
    * `seeds-in-the-wild-jailbreak-prompts`: Real-world jailbreak attempts from TrustAIRLab collected from Discord, Reddit, and other platforms (~1,400 prompts from December 2023). Includes fetch script to download from Hugging Face.
    * `seeds-simsonsun-high-quality-jailbreaks`: Contamination-free jailbreak prompts curated to avoid overlap with training data of popular jailbreak classifiers. Includes two options: Dataset 1 (67 high-quality prompts) and Dataset 2 (2,359 broader coverage prompts). Supports `--dataset 2` flag in fetch script.

### Fixed

* Progress bars now correctly show previous progress when resuming a test with `--resume-file`, instead of starting from zero.

## [0.2.0] - 2025-04-28

### Added

* **Dynamic Attack Framework:**
    * New `attacks/` directory for attack scripts.
    * `spikee test` command now supports `--attack <name>` and `--attack-iterations <N>` to run iterative attack strategies if standard attempts fail.
    * `attacks/sample_attack.py`: Template for creating custom attacks.
    * Built-in attacks: `best_of_n`, `random_suffix_search`, `anti_spotlighting`, `prompt_decomposition_llm`, `prompt_decomposition_dumb`.
* **Judge System for Success Evaluation:**
    * Replaced `--success-criteria` flag with a per-entry judge system.
    * Dataset entries now use `judge_name` (e.g., "canary", "regex", "llm_judge") and `judge_args` (e.g., canary string, regex pattern, LLM criteria) to define success.
    * New `judges/` directory for custom judge scripts.
    * `judges/sample_judge.py`: Template provided via `spikee init`.
    * Built-in judges: `canary`, `regex`. Example LLM judges in workspace.
* **Enhanced Dataset Generation:**
    * Added `payload` field to JSONL output (the raw jailbreak+instruction text).
    * Added `exclude_from_transformations_regex` field (list of regex strings) for finer control over plugin transformations.
    * Plugins can now return a `List[str]` to generate multiple variations per input.
* **New Seed Datasets:**
    * `seeds-wildguardmix-harmful`: For testing harmful content generation (requires fetching external dataset). Uses LLM judge.
    * `seeds-investment-advice`: For testing topical guardrails around financial advice. Includes benign and attack prompts.
    * Updated other seeds like `seeds-cybersec-2025-04`, `seeds-sysmsg-extraction-2025-04`.
* **Results Analysis Improvements:**
    * `spikee results analyze` now correctly handles and groups results from dynamic attacks.
    * Calculates and reports `Initial Success Rate` (without dynamic attack) and `Attack Improvement` (successes achieved only via dynamic attack).
    * Added `response_time` field to results JSONL. For dynamic attacks, this covers the full attack duration.
    * Added `--false-positive-checks <path.jsonl>` option to `spikee results analyze` for calculating precision, recall, F1, and accuracy using results from benign prompts.
* **CLI Enhancements:**
    * `spikee init` now creates `attacks/` and `judges/` directories.
    * `spikee init` supports `--include-builtin [all|plugins|judges|targets|attacks]` to copy built-in modules locally.
    * `spikee list` command now includes `attacks` and `judges`.
    * Added `--tag` option to `spikee generate` and `spikee test` to add custom suffixes to output filenames.
    * Added `--max-retries` flag to `spikee test` (default 3) for rate limit handling.
* **Plugin Interface:**
    * `transform` function in plugins now accepts an optional `exclude_patterns: List[str]` argument.

### Changed

* **Guardrail Target Logic:** Targets intended for guardrail testing must now return `True` if the attack **bypassed** the guardrail (attack successful) and `False` if the attack was **blocked** (attack failed). This standardizes the boolean interpretation across guardrail targets. Built-in guardrail targets have been updated. **This is a potential breaking change if using custom v0.1 guardrail targets.**
* **Success Criteria Deprecated:** Removed the `--success-criteria` flag from `spikee test`. Success is now managed via the Judge system.
* **Internal Refactoring:** Updated `tester.py` and `results.py` to support dynamic attacks and the judge system. `generator.py` updated for new dataset fields and multi-variation plugins.

### Fixed

* Improved handling of file paths and module loading for local vs. built-in components.
* Ensured unique IDs for dynamic attack result entries (`<original_id>-attack`).

## [0.1.0] - 2025-01-27

* Initial release.
* Features: Dataset generation (`spikee generate`), testing (`spikee test`), results analysis (`spikee results analyze`, `convert-to-excel`), basic workspace initialization (`spikee init`), listing components (`spikee list`).
* Support for plugins and targets (local and built-in).
* Success criteria based on `--success-criteria [canary|boolean]`.