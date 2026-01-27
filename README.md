<div align="center">
  <a href="https://github.com/ReversecLabs/spikee">
    <picture>
      <source srcset="/images/logo-dark.png" media="(prefers-color-scheme: dark)">
      <source srcset="/images/logo-light.png" media="(prefers-color-scheme: light)">
      <img src="/images/logo-light.png" alt="Spikee Logo" width="200">
    </picture>
  </a>
  <br>
  <h1>Simple Prompt Injection Kit for Evaluation and Exploitation</h1>

  <p align="center">
    <a href="https://spikee.ai" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">
      <img src="images/link.svg" width="20" height="20" alt="Website" style="vertical-align:middle;">
      &nbsp;Website
    </a>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="https://reversec.com" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">
      <img src="images/link.svg" width="20" height="20" alt="Reversec" style="vertical-align:middle;">
      &nbsp;Reversec
    </a>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="https://discord.gg/hweNfZw5pr" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">
      <img src="images/discord.svg" width="20" height="20" alt="Discord" style="vertical-align:middle;">
      &nbsp;Discord
    </a>
  </p>
</div>

_Version: 0.6.0_


Developed by Reversec Labs, `spikee` is a toolkit for assessing the resilience of LLMs, guardrails, and applications against prompt injection and jailbreaking. Spikee's strength is its modular design, which allows for easy customization of every part of the testing process.

> **0.5.0 MAJOR CHANGE WARNING:** Module interfaces for targets, plugins, attacks, and judges now use an OOP pattern. The legacy function-based hooks still run for now but will be deprecated in v1.0.0 - review the `CHANGELOG.md` entry for 0.5.0 and the sample implementations in the workspace/docs to understand the updated APIs.


---

## Table of Contents

- [Spikee Use Cases](#spikee-use-cases)
- [The Spikee Architecture](#the-spikee-architecture)
- [Documentation](#documentation)
- [Installation](#1-installation)
  - [Local Installation (From Source)](#12-local-installation-from-source)
  - [Local Inference Dependencies](#13-local-inference-dependencies)
- [Core Workflow: A Practical Guide](#2-core-workflow-a-practical-guide)
  - [Step 1: Initialize a Workspace](#step-1-initialize-a-workspace)
  - [Step 2: Explore Available Components](#step-2-explore-available-components)
  - [Step 3: Choose a Scenario and Generate a Dataset](#step-3-choose-a-scenario-and-generate-a-dataset)
    - [Scenario A: Testing a Standalone LLM](#scenario-a-testing-a-standalone-llm)
    - [Scenario B: Testing an LLM Application](#scenario-b-testing-an-llm-application)
    - [Bonus: Including standalone attacks](#bonus-including-standalone-attacks)
  - [Step 4: Test a Target](#step-4-test-a-target)
    - [A. Basic LLM Test](#a-basic-llm-test)
    - [B. Testing a Custom LLM Application](#b-testing-a-custom-llm-application)
    - [C. Resume Options](#c-resume-options)
    - [D. Enhancing Tests with Attacks](#d-enhancing-tests-with-attacks)
    - [E. Testing a Sample of a Large Dataset](#e-testing-a-sample-of-a-large-dataset)
    - [F. Evaluating Guardrails](#f-evaluating-guardrails)
  - [Step 5: Analyze the Results](#step-5-analyze-the-results)
- [Contributing](#3-contributing)
  - [Questions or Feedback?](#questions-or-feedback)

---

## Spikee Use Cases
<div align="center">
    <img src="docs/spikee-usecases.png" width="700px">
</div>

## The Spikee Architecture

Spikee operates in two stages: generating a test dataset, and executing tests against a target using the dataset. Each stage is powered by easy-to-customize Python modules.

<div align="center">
    <img src="docs/spikee-architecture.png" width="700px">
</div>

## Documentation

This README provides a practical guide to the core workflow. For advanced topics, see the detailed documentation:

1.  **[Built-in Seeds, Datasets and Plugins](./docs/01_builtin_datasets_and_plugins.md)**: An overview of all built-in datasets and plugins.
2.  **[Dataset Generation](./docs/02_dataset_generation.md)**: A reference for `spikee generate`.
3.  **[Built-in Targets, Attacks and Judges](./docs/03_builtin_targets_attacks_and_judges.md)**: An overview of all built-in targets, attacks and judges.
4.  **[Creating Custom Targets](./docs/04_custom_targets.md)**: Interact with any LLM, API, or guardrail.
5.  **[Creating Custom Plugins](./docs/05_custom_plugins.md)**: Statically transform and obfuscate payloads.
6.  **[Creating Dynamic Attack Scripts](./docs/06_dynamic_attacks.md)**: Create iterative, adaptive attack logic.
7.  **[Judges: Evaluating Attack Success](./docs/07_judges.md)**: Define custom success criteria for tests.

8.  **[Testing Guardrails Using Spikee](./docs/08_guardrail_testing.md)**: Evaluate guardrail effectiveness and false positive rates.

9.  **[Spikee Results](./docs/09_results.md)**: A reference for `spikee analyze` and an explanation of test reports and performance metrics.
10.  **[Generating Custom Datasets with an LLM](./docs/10_llm_dataset_generation.md)**: Create tailored datasets for specific use cases.
11.  **[Installing Spikee in an Isolated Environment](./docs/11_installing_spikee_in_isolated_enviroments.md)**: Guide to setting up Spikee in a secure, isolated environment.
12. **[Functional Testing Guide](./docs/12_functional_testing.md)**: Run the end-to-end CLI regression suite locally.
---

## 1. Installation

Install `spikee` directly from PyPI.

```bash
pip install spikee
```

To ensure a clean installation when upgrading, use the `--force-reinstall` flag (*this helps a lot removing deprecated files/datasets that would otherwise persist*):
```bash
pip install --upgrade --force-reinstall spikee
```


### 1.2 Local Installation (From Source)

```bash
git clone https://github.com/ReversecLabs/spikee.git
cd spikee
python3 -m venv env
source env/bin/activate
pip install .
```


### 1.3 Local Inference Dependencies

For targets requiring local model inference:

```bash
pip install -r requirements-local-inference.txt
```

---


## 2. Core Workflow: A Practical Guide


### Step 1: Initialize a Workspace

Create a project directory and initialize it. 

`spikee init` will create a folder `workspace` containing built-in datasets, and local sample targets, plugins and attacks.

```bash
mkdir my-spikee-project
cd my-spikee-project
spikee init
```


### Step 2: Explore Available Components

Use `spikee list` to see what seeds, datasets, judges, targets, plugins and attacks are available in your workspace (both local and built-in).

```bash
spikee list seeds
spikee list datasets
spikee list judges
spikee list targets
spikee list plugins
spikee list attacks
```


### Step 3: Choose a Scenario and Generate a Dataset

Your testing scenario determines what kind of testing dataset you need to generate.

> A list of built-in datasets and plugins is available within the **[Built-in Seeds and Datasets](./docs/01_builtin_datasets_and_plugins.md)** documentation and a complete list of dataset generation options is available in the **[Dataset Generation](./docs/02_dataset_generation.md)** documentation.


#### Scenario A: Testing a Standalone LLM
When you test an LLM directly, you control the entire prompt. This is ideal for assessing a model's general resilience to jailbreaks and harmful instructions.

*   **What to Generate:** A *full prompt*, which includes a task (like "Summarize this: <data>"), the data containing the prompt injection or jailbreak, and optionally a system message.
*   **How to Generate:** Use `--format full-prompt` and optionally `--include-system-message`. The `datasets/seeds-cybersec-2026-01` folder provides a great starting point with diverse jailbreaks and attack instructions.

```bash
spikee generate --seed-folder datasets/seeds-cybersec-2026-01 --format full-prompt
```

This will generate the dataset in JSONL format: `datasets/cybersec-2026-01-full-prompt-dataset-<TIMESTAMP>.jsonl`.


#### Scenario B: Testing an LLM Application 
When you test an application (like a chatbot or an email summarizer), the application itself builds the final prompt. Your input is just one part of it, which could be a prompt or data (such as documents/emails).

*   **What to Generate:** Just the *user prompt* or *document* with the attack payload (e.g., the body of an email containing a prompt injection).
*   **How to Generate:** Use `--format user-input` (you can omit as this is the default from v0.4.1).

```bash
spikee generate --seed-folder datasets/seeds-cybersec-2026-01 --format user-input
```

This will generate the dataset in JSONL format: `datasets/cybersec-2026-01-document-dataset-<TIMESTAMP>.jsonl`.


#### Bonus: Including standalone attacks
The `generate` command we saw before composes a dataset by combining documents with jailbreaks and instructions. However, some datasets - such as `seeds-simsonsun-high-quality-jailbreaks` and `in-the-wild-jailbreak-prompts` - contain a static list of ready-to-use attack prompts. To include those in the generated dataset, we use `--include-standalone-inputs`:

```bash
spikee generate --seed-folder datasets/seeds-simsonsun-high-quality-jailbreaks \
                --include-standalone-inputs \
```

#### Bonus: Including Plugins
If static prompts fail, dataset generation can be enhanced with the use of Plugins. These are  scripts that will apply static transformations to payloads during dataset generation, creating multiple iterations of each entry.

```bash
# 1337 (leetspeak) plugin
spikee generate --seed-folder datasets/seeds-cybersec-2026-01 \
                --plugin 1337
```

```bash
# Best of N plugin, with 50 variants per entry
spikee generate --seed-folder datasets/seeds-cybersec-2026-01 \
                --plugin best_of_n \
                --plugin-options "best_of_n:variants=50"
```

### Step 4: Test a Target

`spikee test` runs your dataset against a target. First, rename `.env-example` to `.env` and add any necessary API keys.

> A list of built-in targets is available within the **[Built-in Targets, Plugins, Attacks and Judges](./docs/04_builtin_targets_plugins_attacks_and_judges.md)** documentation.

#### A. Basic LLM Test
This command tests gpt-4o-mini via the OpenAI API using the dataset generated in Scenario A (require `OPENAI_API_KEY` in `.env`).

```bash
spikee test --dataset datasets/cybersec-2026-01-full-prompt-dataset-*.jsonl \
            --target openai_api \
            --target-options gpt-4o-mini
```

> **How is attack success determined? With Judges.**
>
> The `cybersec-2026-01` dataset contains attacks whose success can be verified automatically by searching for specific "canary" words or matching regular expressions in the response (such as the presence of a *Markdown image*).
>
> For more complex goals, like checking for harmful content or policy violations, Spikee can use more complex **Judges**. These are Python modules that evaluate the target's response. We include simple LLM-based judges that can assess if a response meets a given criteria. See the **[Judges documentation](./docs/07_judges.md)** to learn more.


#### B. Testing a Custom LLM Application
To test an LLM application, you must create a custom **Target script**. This Python script, placed in the `targets/` directory in your workspace, tells Spikee how to send data to the application and receive its response. For details, see the **[Creating Custom Targets](./docs/04_custom_targets.md)** guide.

```bash
# Test a custom email application using malicious documents and your custom target
spikee test --dataset datasets/llm-mailbox-document-dataset-*.jsonl \
            --target llm_mailbox
```

> Especially when testing LLM applications, it's useful to create a custom dataset tailored to the specific use case. In the sample case of the LLM Webmail application, we create a custom dataset stating from `cybersec-2026-01`, that only focusses on testing exfiltration of confidential information via markdown images. Check this tutorial for more information: https://labs.reversec.com/posts/2025/01/spikee-testing-llm-applications-for-prompt-injection


#### C. Resume Options
Spikee can resume interrupted runs:

- `--resume-file <file>`  
  Resume explicitly from a given results file. (This flag will only function with a single dataset file.)

- `--auto-resume`  
  Automatically resume from the latest matching results file without prompting. If no file is found, start a new run.

- `--no-auto-resume`  
  Always creates a new results file. This will disable the interactive resume prompt and ignore existing results files.

**Default behavior**  
- In interactive TTY: Spikee searches for matching results files and prompts you to select a results file to resume.  
- In non-TTY: Spikee does not auto-resume unless `--auto-resume` is set.


#### D. Multi-File Options
Spikee allows for multiple dataset files to be provided in a single test run using the `--dataset` and `--dataset-folder` flags.
* Dataset files processed sequentially in the provided order (folders are scanned and included in alphabetical order), generating a results file for each dataset.
* The `--dataset-folder` flag will scan the specified folder for all JSONL files and include them in the test run (NB, beware no checks are made to ensure the included files are dataset files).
* Any supplied flags will be applied to all datasets.
* Multi-file is not compatible with single dataset flags (e.g., `--resume-file`). (NB, other resume options are compatible, default interactive resume prompt will occur prior to each dataset file being processed).

```bash
spikee test --dataset datasets/cybersec-2026-01-full-prompt-dataset-*.jsonl \
            --dataset datasets/simsonsun-high-quality-jailbreaks-full-prompt-dataset-*.jsonl \
            --target openai_api \
            --target-options gpt-4o-mini
```

```bash
spikee test --dataset-folder datasets/
            --target openai_api \
            --target-options gpt-4o-mini
```


#### E. Enhancing Tests with Attacks
If static prompts fail, use `--attack` to run iterative scripts that modifies the prompt/documents until they succeed (or run out of iterations).

```bash
# Best of N attack
spikee test --dataset datasets/dataset-name.jsonl \
            --target openai_api \
            --attack best_of_n --attack-iterations 25
```

```bash
# Anti spotlighting attack
spikee test --dataset datasets/dataset-name.jsonl \
            --target openai_api \
            --attack anti_spotlighting --attack-iterations 50
```

Some attacks, like `prompt decomposition` support options, such as which LLM to use to generate attack prompt variations:
```bash
spikee test --dataset datasets/dataset-name.jsonl \
            --target openai_api \
            --attack prompt_decomposition --attack-iterations 50 -attack-options 'prompt_decomposition:variants=15;mode=ollama-llama3.2'
```

```bash
# LLM Jailbreaker attack
spikee test --dataset datasets/dataset-name.jsonl \
            --target openai_api \
            --attack llm_jailbreaker --attack-iterations 10
```

```bash
# LLM Multi-language Jailbreaker attack
spikee test --dataset datasets/dataset-name.jsonl \
            --target openai_api \
            --attack llm_multi_language_jailbreaker --attack-iterations 10
```

> See **[Built-in Targets, Plugins, Attacks and Judges](./docs/03_builtin_targets_attacks_and_judges.md)** for a list of built-in attacks and their alternate plugin versions, and **[Creating Dynamic Attack Scripts](./docs/06_dynamic_attacks.md)** for information on writing attacks.


#### F. Testing a Sample of a Large Dataset
For large datasets, or when operating under time and cost constraints, you can test a random subset of the dataset using the `--sample` flag.

By default, Spikee uses a static seed for sampling. This means that running the same command multiple times will always select the **same random sample**, ensuring your tests are reproducible. This is useful for regression testing.

```bash
# Test a reproducible 15% sample of a large dataset.
# This will select the same 15% of entries every time you run it.
spikee test --dataset datasets/large-dataset.jsonl \
            --target openai_api \
            --sample 0.15
```

If you need a different sample for each run, or want to use your own seed for reproducibility across different machines or setups, you can use the `--sample-seed` flag.

```bash
# Use a custom seed for a different reproducible sample
spikee test --dataset datasets/large-dataset.jsonl \
            --target openai_api \
            --sample 0.1 \
            --sample-seed 123

# Use a truly random sample on each run
spikee test --dataset datasets/large-dataset.jsonl \
            --target openai_api \
            --sample 0.1 \
            --sample-seed random
```

#### G. Evaluating Guardrails
When you're testing an LLM application, you're automatically testing any guardrail that the developers of the application have applied. However, sometimes you might want to test individual guardrails in isolation. 

Further information on evaluating guardrails can be found in the **[Guardrail Testing](./docs/08_guardrail_testing.md)** documentation.

**1. Testing a Prompt Injection Guardrail:**
To test a guardrail's ability to block general jailbreaks, you could use a broad dataset like `in-the-wild-jailbreak-prompts`, or a more high-quality, focussed one like `seeds-simsonsun-high-quality-jailbreaks`.

```bash
# Test Meta's Prompt Guard against jailbreaks
spikee generate --seed-folder datasets/seeds-simsonsun-high-quality-jailbreaks \
                --include-standalone-inputs \

spikee test --dataset datasets/simsonsun-high-quality-jailbreaks-*.jsonl \
            --target prompt_guard_jailbreak
```

The output of this will tell you whether a particular prompt in the dataset bypassed the guardrail. *This doesn't mean the jailbreak would succeed against an LLM, but simply that it would not be blocked by a guardrail*.

**2. Testing a Topical Guardrail:**
To test a guardrail that blocks specific topics (like financial advice), use a purpose-built dataset, like the sample one that can be generated from these seeds: `seeds-investment-advice`.

```bash
spikee generate --seed-folder datasets/seeds-investment-advice \
                --include-standalone-inputs \
                --include-system-message \
```

Notice that here we use `--format document`, as we just want to generate the raw prompts/queries from the seed folder, we don't want to add additional prompts. Also notice that we use `--include-system-message`, as the topical guardrail will use this to determine whether the input aligns with the system_message rules.

```bash
spikee test --dataset datasets/investment-advice-document-sys-dataset-TIMESTAMP.jsonl \
            --target nvidia_nemoguard_topic_control
```

#### H. Multi-Turn Attacks
Spikee allows for multi-turn attacks, when an compatible target, attack script and dataset is used. This enabled Spikee to assess multi-turn applications, such as chatbots, against multi-turn prompt injection attacks. 

See the following documentation to implement custom [targets](./docs/04_custom_targets.md#multi-turn-dynamic-targets) and [attacks](./docs/06_dynamic_attacks.md#multi-turn-dynamic-attacks) and the following on generating multi-turn [datasets](./docs/02_dataset_generation.md#multi-turn-datasets).

The following command will perform a `crescendo` multi-turn attack using an instructional multi-turn dataset, against an `LLMBank` target:

```bash
spikee test --dataset datasets/multi-turn-dataset-*.jsonl \
            --target LLMBank \
            --attack crescendo
            --attack-options 'model=openai-gpt-4o-mini'
```

### Step 5: Analyze the Results

Use `spikee results analyze` to get a statistical summary of the test run.

```bash
# Analyze the most recent results file for the openai_api target
spikee results analyze --result-file results/results_openai_api-gpt-4o-mini_*.jsonl
```
This command provides an overview of the success rate and detailed breakdowns by attack type, helping you identify specific weaknesses.

Further information on analyzing results can be found in the **[Spikee Results](./docs/09_results.md)** documentation.

---

## 3. Contributing

Contributions are welcome. Please feel free to submit bug fixes, new modules (Targets, Plugins, Attacks, Judges), or dataset seeds via GitHub pull requests.

### Questions or Feedback?

File an issue on the [GitHub repository](https://github.com/ReversecLabs/spikee).
