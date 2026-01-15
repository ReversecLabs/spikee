# Targets
Spikee includes a variety of built-in and sample targets, an overview of each can be seen below. 

Available targets can be listed at any time with `spikee list targets`.

## Built-in Targets
Built-in targets focus on several common LLM providers, and will require you to rename `.env-example` to `.env` and add any necessary API keys - these are located within the `spikee/targets/` folder. 

**Common Providers**
* `aws_bedrock_api`: AWS Bedrock API (Requires AWS credentials)
* `azure_api`: Azure OpenAI Service API (Requires `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`)
* `deepseek_api`: Deepseek API (Requires `DEEPSEEK_API_KEY`)
* `google_api`: Google Gen AI API (Requires `GOOGLE_API_KEY`)
* `groq_api`: Groq API (Requires `GROQ_API_KEY`)
* `openai_api`: OpenAI API (Requires `OPENAI_API_KEY`)
* `togetherai_api`: TogetherAI API (Requires `TOGETHERAI_API_KEY`)

**Local Models**
* `llamacpp_api`: Modified adaptation of the OpenAI API for LLaMA CCP Servers. 
    * Default: `http://localhost:8080/`
    * Custom: `--target-options http://your-llamacpp-server:port/`
* `ollama_api`: Ollama API (Requires Ollama running locally: `http://localhost:11434/`)


**Other**
* `aws_bedrock_guardrail` **(Legacy Target)**: Assess AWS Bedrock Guardrails
* `az_ai_content_safety_harmful` **(Legacy Target)**: Assess Azure AI Content Safety Harm Categories
* `az_prompt_shields_document_analysis` **(Legacy Target)**: Assess Azure Prompt Shields Document Analysis
* `az_prompt_shields_prompt_analysis` **(Legacy Target)**: Assess Azure Prompt Shields Prompt Analysis

## Usage Example
```bash
# Test an AWS Bedrock target using the built-in target module
spikee test --dataset datasets/cybersec-2025-04.jsonl \
            --target aws_bedrock_api
```

```bash
# Test an LLaMA CPP Server running on localhost port 9090
spikee test --dataset datasets/cybersec-2025-04.jsonl \
            --target llamacpp_api \
            --target-options http://localhost:9090/
```


## Sample Targets
Sample targets are provided within the `workspace/targets/` folder, created by running `spikee init`. These demonstrate how to write custom targets and can be easily modified to assess an LLM application of your choice.

* `llm_mailbox`: Sample target for email summarisation application tutorial https://labs.reversec.com/posts/2025/01/spikee-testing-llm-applications-for-prompt-injection
* `test_chatbot` and `simple_test_chatbot`: Sample Multi-Turn targets for the Spikee Test Chatbot (https://github.com/ReversecLabs/spikee-test-chatbot), demonstrating `multi_target` and `simple_multi_target` usage.
* `sample_pdf_request_target`: Sample target - Sends a POST request containing a PDF to a fictional application.
* `sample_target_legacy` **(Legacy Target)**: Sample legacy target - Returns a mock message.
* `sample_target`: Sample target - Sends a GET request to a fictional application, demonstrating options and advanced guardrail and error handling.


# Built-in Attacks
Spikee includes several built-in dynamic attacks, that will iteratively modify prompts/documents until they succeed (or run out of iterations). These are located within the `spikee/attacks/` folder, and can be listed at any time with `spikee list attacks`.

You can customize the behavior of attacks using the following command-line options:
* `--attack-iterations`: Specifies the maximum number of iterations for each attack (default: 1000).
* `--attack-options`: Passes a single string option to the attack script for custom behavior (e.g., `"mode=aggressive"`).


The following list provides an overview of each attack, further information on each attack can be found within each attack's file.

**Single-Turn:**
* `anti_spotlighting`: Assess spotlighting vulnerabilities by sequentially trying variations of delimiter-based attacks.
* `best_of_n`: Implements ["Best-of-N Jailbreaking" John Hughes et al., 2024](https://arxiv.org/html/2412.03556v1#A1) to apply character scrambling, random capitalization, and character noising.
* `prompt_decomposition`: Decomposes a prompt into chunks and generates shuffled variations.
    * Options: 
        `modes` (LLM model to apply, default: dumb),
        `variants` (number of variations to generate, default: 50).
* `random_suffix_attack`: Implements [Random Suffix Search](https://arxiv.org/abs/2404.02151) techniques, which appends random suffixes to the prompt to bypass filters.

**Multi-Turn:**
* `multi_turn`: Sequentially sends a predefined list of user prompts to the target LLM, from a simplistic multi-turn dataset. 
* `crescendo`: Implements the [Crescendo Attack](https://arxiv.org/abs/2404.01833). This is a simple multi-turn jailbreak that leverages an LLM Agent to prompt the target application with seemingly benign prompts, but gradually escalates the conversation by referencing the model's replies progressively leading to a successful jailbreak.

(NB, See [Multi-Turn Datasets](./02_dataset_generation.md#multi-turn-datasets) for more information.)

## Usage Example
```bash
# Test an AWS Bedrock target using the prompt decomposition attack with 500 iterations and custom options
spikee test --dataset datasets/cybersec-2025-04.jsonl \
            --target aws_bedrock_api \
            --attack prompt_decomposition \
            --attack-iterations 500 \
            --attack-options "modes=dumb;variants=50"
```

# Built-in Judges
Spikee includes several built-in judges to evaluate LLM responses, located within the `spikee/judges/` folder. These can be listed at any time with `spikee list judges`.

## Basic Judges
These are basic judges that evaluate responses based on simple criteria.

* `canary`: Checks if a predefined canary string is present in the response.
* `regex`: Uses regular expressions to identify specific patterns in the response.

## LLM Judges
For some test cases, success cannot be determined by a simple keyword or pattern. For instance, did the model's response contain harmful advice, or did it refuse to answer a question on a restricted topic?

LLM-based judges address this by using a separate LLM to evaluate the target's response against a natural language criterion.

* `llm_judge_harmful`: LLM judge to evaluate whether the target LLMs response complied with a potentially harmful user prompt.
* `llm_judge_output_criteria`: LLM judge to evaluate whether the target LLMs response meets specific success criteria defined in `judge_args`.

You can specify the LLM model used with the `--judge-options` flag. It currently supports the following providers, replace `<model_name>` with the desired model from that provider. (Providers may require API keys to be configured within your `.env` file):
* `offline`: Mock judge, for restrictive environments. See [re-judging](<./07_judges.md#Re-judging>) and [isolated environments](./11_installing_spikee_in_isolated_environments.md) documentation for more information.
* `openai-<model_name>`: OpenAI API (e.g., `openai-gpt-4o-mini`)
* `ollama-<model_name>`: Ollama API (e.g., `ollama-gemma3`)
* `bedrock-<model_name>`: AWS Bedrock API (e.g., `bedrock-us.anthropic.claude-3-7-sonnet-20250219-v1:0`)
* `together-<model_name>`: TogetherAI API (e.g., `together-gemma2-8b`)
* `llamacpp-server` or `llamacpp-server-<port>`: Local LLaMA CPP Server (e.g., `llamcpp-server` defaults to port 8080, or `llamacpp-server-9090` for port 9090)

## Usage Example
```bash
# Use an offline judge, allowing for later re-judging
spikee test --dataset datasets/cybersec-2025-04.jsonl \
            --target aws_bedrock_api \
            --judge-options offline
```

```bash
# Modify entries to use llm_judge_harmful, with the an OpenAI's gpt-4o-mini model for judging
spikee test --dataset datasets/cybersec-2025-04.jsonl \
            --target aws_bedrock_api \
            --judge llm_judge_harmful \
            --judge-options openai-gpt-4o-mini
```
