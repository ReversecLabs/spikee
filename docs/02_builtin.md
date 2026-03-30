# Built-In Seeds and Modules
Spikee comes with a variety of built-in seeds and modules (e.g., targets, judges, plugins, attacks). 

Jump to Links:
- [Built-in Seeds](#built-in-seeds)
- [Built-in Targets](#built-in-targets)
- [Built-in Judges](#built-in-judges)
- [Built-in Plugins](#built-in-plugins)
- [Built-in Attacks](#built-in-attacks)

## Built-in Seeds
Spikee comes with a variety of built-in seeds, each designed for a specific testing purpose. These seeds are located in the `datasets/` directory after you run `spikee init`. You can list them at any time with `spikee list seeds`.

| Seed | Source | Type | Description |
|------|-------------|-------------|-------------|
| `seeds-cybersec-2026-01` | Reversec | Cybersecurity | A general-purpose dataset for testing prompt injection and cybersecurity harms. It focuses on common attack goals seen in web application security, such as data exfiltration, cross-site scripting (XSS), and resource exhaustion. |
| `seeds-harmful-instructions-only` | Reversec | Objectives | Specifically designed for attacks using **LLM Agents**, such as **Crescendo** and **LLM-Jailbreaker**. These attacks require a instruction (objective) to generate their own attack vectors dynamically. Contains harmful instructions in `instructions.jsonl`, while leaving jailbreaks and user inputs as empty placeholders. |
| `seeds-simsonsun-high-quality-jailbreaks` | External | Jailbreaks | A high-quality set of contamination-free jailbreak prompts, specifically curated to avoid overlap with the training data of many common safety classifiers. |
| `seeds-in-the-wild-jailbreak-prompts` | External | Jailbreaks | Contains approximately 1,400 real-world jailbreak prompts collected from public sources like Discord and Reddit (filtered from the TrustAIRLab dataset). Ideal for testing a target's resilience against known, publicly available jailbreaks. |
| `seeds-wildguardmix-harmful` | External | Harmful | A dataset for testing harmful content generation. The prompts are sourced from the WildGuard-Mix dataset. |
| `seeds-wildguardmix-harmful-fp` | External | Harmful (FP) | A companion dataset to `seeds-wildguardmix-harmful`, containing benign (harmless) prompts. |
| `seeds-toxic-chat` | External | Harmful | A dataset for testing toxic prompts, filtered from 10K user prompts collected from the Vicuna online demo. |
| `seeds-investment-advice` | Reversec | Topical Guardrails | Designed to test topical guardrails that are supposed to block personal financial or investment advice. It includes both malicious instructions and standalone attack prompts. |
| `seeds-investment-advice-fp` | Reversec | Topical Guardrails (FP) | A companion dataset to `seeds-investment-advice`, containing benign (harmless) queries about financial topics. |
| `seeds-sysmsg-extraction-2025-04` | Reversec | System Prompt Extraction | Specifically designed to test for system prompt extraction. The instructions and judges are tailored to detect if the target model leaks its own system prompt or initial instructions. |
| `seeds-llm-mailbox` | Reversec | Tutorial | An example seed tailored for testing an email summarization feature. The documents are sample emails, and the instructions are designed to test for vulnerabilities in that specific context. See the associated [blog post](https://labs.reversec.com/posts/2025/01/spikee-testing-llm-applications-for-prompt-injection) for a detailed walkthrough. |
| `seeds-empty` | Reversec | Utility | An empty template folder. It contains empty `documents.jsonl`, `jailbreaks.jsonl`, and `instructions.jsonl` files. This is the recommended starting point when creating a new dataset from scratch, especially for standalone attacks. |
| `seeds-mini-test` | Reversec | Utility | A very small set of examples for quick, functional testing of Spikee itself. Use this to verify your setup or to test a new custom target or plugin without running a large number of tests. |

> FP datasets are intended for use with the `--false-positive-checks` flag to measure how often a guardrail incorrectly blocks legitimate prompts when evaluating harmful content filters.
>
> External datasets require you to run a fetch script to download the prompts. See the `README.md` inside each seed folder for instructions. Some of these use an LLM judge by default, which will be specified in the seed's README.

** Usage Example**
```bash
spikee generate --seed ./seeds-cybersec-2026-01
```

## Built-in Targets

Spikee includes a variety of built-in and sample targets, which can be listed at any time with `spikee list targets`.

**Built-in targets** focus on several common LLM providers, and will require you to rename `.env-example` to `.env` and add any necessary API keys - these are located within the `spikee/targets/` folder. 

| Target | Type | Description |
|--------|------|-------------|
| `llm_provider` | Provider | Generic LLM target for supported LLM providers (e.g., openai, bedrock, google, ollama, e.t.c.) ([See Docs](./03_llm_providers.md)) |
| `aws_bedrock_guardrail` | Guardrails | Assess AWS Bedrock Guardrails |
| `az_ai_content_safety_harmful` | Guardrails | Assess Azure AI Content Safety Harm Categories |
| `az_prompt_shields_document_analysis` | Guardrails | Assess Azure Prompt Shields Document Analysis  |
| `az_prompt_shields_prompt_analysis` | Guardrails | Assess Azure Prompt Shields Prompt Analysis |


**Sample targets** are provided within the `workspace/targets/` folder - created by running `spikee init`. These demonstrate how to write custom targets and can be easily modified to assess an LLM application of your choice.

| Target | Type | Description |
|--------|------|-------------|
| `sample_target` | Single-Turn | Sends a GET request to a fictional application, demonstrating options and advanced guardrail and error handling. |
| `sample_target_legacy` | Single-Turn (Legacy) | Returns a mock message. This is a legacy target, demonstrating the older target format. |
| `sample_pdf_request_target` | Single-Turn | Sends a POST request containing a PDF to a fictional application. |
| `test_chatbot` | Multi-Turn | Sends requests to [Spikee Test Chatbot](https://github.com/ReversecLabs/spikee-test-chatbot) |
| `simple_test_chatbot` | Multi-Turn | Implements the simple multi-turn target, and sends requests to [Spikee Test Chatbot](https://github.com/ReversecLabs/spikee-test-chatbot) |
|`llm_mailbox` | Single-Turn | Sample target for email summarisation application [tutorial](https://labs.reversec.com/posts/2025/01/spikee-testing-llm-applications-for-prompt-injection) |

**Usage Example**
```bash
spikee test --dataset datasets/cybersec-2026-01.jsonl \
            --target llm_provider \
            --target-options "bedrock/claude45-haiku"
```


## Built-in Judges

Spikee includes several built-in judges to evaluate LLM responses, located within the `spikee/judges/` and `workspace/judges/` folders. These can be listed at any time with `spikee list judges`.

**Basic Judges**
These evaluate responses based on simple criteria.

* `canary`: Checks if a predefined canary string is present in the response.
* `regex`: Uses regular expressions to identify specific patterns in the response.

**LLM Judges**
Some test cases, success cannot be determined by a simple keyword or pattern. For instance, did the model's response contain harmful advice, or did it refuse to answer a question on a restricted topic?

LLM-based judges address this by using a separate LLM to evaluate the target's response against a natural language criterion.

* `llm_judge_harmful`: LLM judge to evaluate whether the target LLMs response complied with a potentially harmful user prompt.
* `llm_judge_objective`: LLM judge to evaluate whether the target LLMs response meets a specific input objective.
* `llm_judge_output_criteria`: LLM judge to evaluate whether the target LLMs response meets specific success criteria defined in `judge_args`.

The LLM Agent model can be specified using the `--judge-options` flag. See **[LLM Providers](./03_llm_providers.md)** for a complete list of supported models, prefixes, and examples. Some common examples include 
* `offline`: Mock judge, for restrictive environments. See [re-judging](<./08_judges.md#Re-judging>) and [isolated environments](./12_installing_spikee_in_isolated_environments.md) documentation for more information.
* `bedrock/<model_name>`: AWS Bedrock API (e.g., `bedrock/claude45-haiku`)
* `openai/<model_name>`: OpenAI API (e.g., `openai/gpt-4o-mini`)
* `google/<model_name>`: Google Gen AI API (e.g., `google/gemini-2.5-flash`)

**Usage Example**
```bash
# Use an offline judge, allowing for later re-judging
spikee test --dataset datasets/cybersec-2026-01.jsonl \
            --target llm_provider \
            --target-options "bedrock/claude45-haiku" \
            --judge-options offline
```

## Built-in Plugins
Spikee includes several build-in plugins, that can be leveraged to enhance dataset generation. These are scripts that will apply static transformations to payloads during dataset generation, and can create multiple iterations of each entry. Built-in plugins are located in the `spikee/plugins/` directory, local plugins are located in the `plugins/` directory within your workspace. You can list them at any time with `spikee list plugins`.

The following list provides an overview of each build-in plugin, further information on each plugin can be found within the plugin file.

**Key**:
- Basic: Simple text transformations.
- Attack-Based: Plugins based on dynamic attack techniques, but have been adapted to work as static transformations during dataset generation.
- LLM: Plugins that leverage an LLM agent to generate variations of the input based on a specific attack strategy or objective.

| Plugin | Type | Description | Options |
|--------|------|-------------|---------|
| `1337` | Basic | Transforms text into "leet speak" by replacing certain letters with numbers or symbols. | N/A |
| `ascii_smuggler` | Basic | Transforms ASCII text into a series of Unicode rags that are generally invisible to most UI elements (bypassing content filters). | N/A |
| `base64` | Basic | Encodes text using Base64 encoding. | N/A |
| `ceasar` | Basic | Applies a Caesar cipher to the text, shifting letters by a specified number of positions. | `shift` (number of positions to shift, default: 3) |
| `flip` | Basic | Applies a flip attack to obfuscate text:<br> - FWO: Flip Word Order<br> - FCW: Flip Chars in Word<br> - FCS: Flip Chars in Sentence | `mode` (the flip mode to apply, default: `FWO`) |
| `google_translate` | Basic | Translates text to another language using google translate. | `source-lang` (language code for source language, default: `en`)<br> `target-lang` (language code for target language, default: `zh-cn`) |
| `opus_translator` | Basic | Translates text to another language using local OPUS-MT models. | `source` (source language code, default: `en`)<br> `targets` (target language(s), default: `zh`)<br> `quality` (translation quality, default: 1)<br> `device` (cpu or gpu, default: auto-detect)<br> `cache_dir` (directory to cache ML models, optional) |
| `hex` | Basic | Encodes text into its hexadecimal representation. | N/A |
| `mask` | Basic | Masks high-risk words in the text with random character sequences, while providing a suffix that maps the masks back to the original words. | `advanced` (if true, creates multiple masks for longer words)<br> `advanced-split` (the number of characters per mask chunk for the advanced option, default: 6) | |
| `morse` | Basic | Encodes text into Morse code. | N/A |
| `splat` | Basic | Obfuscates the text using splat-based techniques (e.g., asterisks '*', special characters, and spacing tricks), to bypass basic filters. | `character` (the character to use for splatting, default: `*`)<br> `insert_rand` (probability of inserting a splat within words, default: 0.6)<br> `pad_rand` (probability of padding words with splats, default: 0.4) |
| `anti_spotlighting` | Attack-Based | Generates variations of delimiter-based attacks to test LLM applications against spotlighting vulnerabilities. | `variants` (number of variations to generate, default: 50) |
| `best_of_n` | Attack-Based | Implements ["Best-of-N Jailbreaking" John Hughes et al., 2024](https://arxiv.org/html/2412.03556v1#A1) to apply character scrambling, random capitalization, and character noising. | `variants` (number of variations to generate, default: 50) |
| `prompt_decomposition` | Attack-Based | Decomposes a prompt into chunks and generates shuffled variations. | `modes` (LLM model to apply, default: dumb)<br> `variants` (number of variations to generate, default: 50) |
| `shortener` | LLM | Uses an LLM to shorten the text to a specified maximum length while retaining key details. | `max_length` (the maximum length for the shortened text, default: 256) |
| `llm_jailbreaker` | LLM | Uses an LLM to iteratively generate jailbreak attacks against the target. | `model` (The LLM model to use for generating attacks, default: `model=openai/gpt-4o`)<br> `variants` (number of variations to generate, default: 5) |
| `llm_multi_language_jailbreaker` | LLM | Generates jailbreak attempts using different languages, focusing on low-resource languages. | `model` (The LLM model to use for generating attacks, default: `model=openai/gpt-4o`)<br> `variants` (number of variations to generate, default: 5) |
| `llm_poetry_jailbreaker` | LLM | Generates jailbreak attempts in the form of poetry or rhymes. | `model` (The LLM model to use for generating attacks, default: `model=openai/gpt-4o`)<br> `variants` (number of variations to generate, default: 5) |
| `rag_poisoner` | LLM | Injects fake RAG context that appears to be legitimate document snippets supporting the attack objective. | `model` (The LLM model to use for generating attacks, default: `model=openai/gpt-4o`)<br> `variants` (number of variations to generate, default: 5) |

**Usage Example**
```bash
spikee generate --seed ./seeds-cybersec-2026-01 \
                --plugin best_of_n google_translate|base64 \
                --plugin-options "best_of_n:variants=5;google_translate:source-lang=en"
```

## Built-in Attacks
Spikee includes several built-in dynamic attacks, that will iteratively modify prompts/documents until they succeed (or run out of iterations). These are located within the `spikee/attacks/` folder, and can be listed at any time with `spikee list attacks`.

You can customize the behavior of attacks using the following command-line options:
* `--attack-iterations`: Specifies the maximum number of iterations for each attack (default: 1000).
* `--attack-options`: Passes a single string option to the attack script for custom behavior (e.g., `"mode=aggressive"`).

| Attack | Type | Description | Additional Options |
|--------|------|-------------|---------|
| `anti_spotlighting` | Standard | Assess spotlighting vulnerabilities by sequentially trying variations of delimiter-based attacks. | N/A |
| `best_of_n` | Standard | Implements ["Best-of-N Jailbreaking" John Hughes et al., 2024](https://arxiv.org/html/2412.03556v1#A1) to apply character scrambling, random capitalization, and character noising. | N/A |
| `prompt_decomposition` | Standard | Decomposes a prompt into chunks and generates shuffled variations. | `modes` (LLM model to apply, default: dumb)<br> `variants` (number of variations to generate, default: 50) |
| `random_suffix_attack` | Standard | Implements [Random Suffix Search](https://arxiv.org/abs/2404.02151) techniques, which appends random suffixes to the prompt to bypass filters. | N/A |
| `llm_jailbreaker` | LLM-Driven | Uses an LLM to iteratively generate jailbreak attacks against the target. | `model` (The LLM model to use for generating attacks, e.g., `model=openai/gpt-4o`) |
| `llm_multi_language_jailbreaker` | LLM-Driven | Generates jailbreak attempts using different languages, focusing on low-resource languages. | `model` (The LLM model to use for generating attacks) |
| `llm_poetry_jailbreaker` | LLM-Driven | Generates jailbreak attempts in the form of poetry or rhymes. | `model` (The LLM model to use for generating attacks) |
| `rag_poisoner` | LLM-Driven | Injects fake RAG context that appears to be legitimate document snippets supporting the attack objective. | `model` (The LLM model to use for generating attacks) |
| `multi_turn` | Simple Multi-Turn | Sequentially sends a predefined list of user prompts to the target LLM, from a simplistic multi-turn dataset. | N/A |
| `crescendo` | Instructional Multi-Turn | Implements the [Crescendo Attack](https://arxiv.org/abs/2404.01833). This is a simple multi-turn jailbreak that leverages an LLM Agent to prompt the target application with seemingly benign prompts, but gradually escalates the conversation by referencing the model's replies progressively leading to a successful jailbreak. | N/A |
| `echo_chamber` | Instructional Multi-Turn | Implements the [Echo Chamber Attack](https://arxiv.org/pdf/2601.05742). This multi-turn attack uses an LLM Agent to create a feedback loop, where the model's own responses are fed back into itself in order to bypass guardrails and achieve jailbreaks. | N/A |
| `goat` | Instructional Multi-Turn | Implements the [GOAT Attack](https://arxiv.org/abs/2404.02151). This multi-turn attack uses an LLM, acting as an automated red teaming agent, that can implement a range of adversarial prompting and jailbreaking techniques to achieve an objective. | See file for target specific configuration using `APPLICATION_CONFIG` and `APPLICATION_GUARDRAILS`. |

**Usage Example**
```bash
spikee test --dataset datasets/dataset-name.jsonl \
            --target demo_llm_application \
            --attack crescendo \
            --attack-options 'max-turns=5,model=bedrock/deepseek-v3' \
            --attack-only

```