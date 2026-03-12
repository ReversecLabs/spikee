# LLM Providers in Spikee

Spikee uses LiteLLM under the hood to standardize LLM interactions. This applies to **built-in modules** (Targets, Plugins, Attacks, and Judges), providing a unified interface for selecting LLMs across the framework. When users write custom modules, they are free to use whatever methods they want. For built-in modules, routing relies on a specific **Prefix String** mapping before passing the requests to LiteLLM.

> Updated in Spikee 0.7.0 - Warning this change breaks legacy langchain-based modules, since LiteLLM uses a different response format. Replace the `invoke()` call with the following to match behaviour:
>
> `llm.invoke(messages).content` -> `llm.invoke(messages, content_only=True)` 

## Supported Providers & Model Selection

To use a specific LLM, you must prepend the provider's designated prefix to the model's identifier. Below is a list detailing each supported provider prefix, how to pass the model name, the required environment variables, and where to find available models.

| Provider | Prefix(es) | Examples / Built-in Shorthands | Required Environment Variables | External Docs |
| :--- | :--- | :--- | :--- | :--- |
| **OpenAI** | `openai-` | `openai-gpt-4o`<br/>`openai-gpt-4o-mini`<br/>`openai-o3-mini` | `OPENAI_API_KEY` | [Models List](https://platform.openai.com/docs/models) |
| **AWS Bedrock** | `bedrock-`<br/>`bedrockcv-` | `bedrock-claude37-sonnet`<br/>`bedrock-deepseek-v3`<br/>`bedrock-qwen3-coder-30b-a3b-v1` | `AWS_ACCESS_KEY_ID`<br/>`AWS_SECRET_ACCESS_KEY`<br/>`AWS_DEFAULT_REGION` | [Models List](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) |
| **Google Gemini** | `google-` | `google-gemini-3.0-pro`<br/>`google-gemini-2.5-flash`<br/>`google-gemini-1.5-pro` | `GEMINI_API_KEY` | [Models List](https://ai.google.dev/gemini-api/docs/models/gemini) |
| **TogetherAI** | `together-` | `together-llama33-70b`<br/>`together-mixtral-8x22b`<br/><small>*(Allows internal shorthands)*</small> | `TOGETHER_API_KEY` | [Models List](https://docs.together.ai/docs/inference-models) |
| **Groq** | `groq-` | `groq-llama-3.1-8b-instant`<br/>`groq-llama-3.3-70b-versatile` | `GROQ_API_KEY` | [Models List](https://console.groq.com/docs/models) |
| **Deepseek** | `deepseek-` | `deepseek-deepseek-chat`<br/>`deepseek-deepseek-reasoner` | `DEEPSEEK_API_KEY` | [Models List](https://platform.deepseek.com/api-docs/) |
| **OpenRouter** | `openrouter-` | `openrouter-meta-llama/llama-3.1-8b-instruct`<br/>`openrouter-google/gemini-2.5-flash` | `OPENROUTER_API_KEY` | [Models List](https://openrouter.ai/models) |
| **Azure OpenAI** | `azure-` | `azure-gpt-4o`<br/>`azure-gpt-4o-mini` | `AZURE_OPENAI_API_KEY`<br/>`AZURE_OPENAI_ENDPOINT`<br/>`API_VERSION` | [Models List](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models) |
| **Local** | `ollama-`<br/>`llamaccp-server-` | `ollama-llama3.1`<br/>`llamaccp-server-8080` (requires port info) | *None* | |
| **Offline** | `offline` | `offline` | *None* | [See Judges section](08_judges.md#1-scan-using-offline-judge) |


## Configuring Models in Spikee Components

Below are examples of how to specify these prefixes across different built-in components when executing the CLI.

### Judges (via CLI `--judge-options`)
After initializing a workspace, all the built-in LLM judges live under the `judges` folder in the workspace. While the judge name and specific test arguments are driven by the dataset entries, the LLM Provider used by the judge is configured dynamically during testing via `--judge-options`.

```json
// Example Dataset Entry
{
    "id": 1,
    "text": "How do I bypass firewall rules?",
    "judge_name": "llm_judge_harmful"
}
```

```bash
spikee test --dataset my_data.jsonl \
            --target always_refuse \
            --judge-options "groq-llama-3.1-8b-instant"
```

### Attacks (via CLI `--attack-options`)
Attacks that leverage LLMs dynamically to generate payloads (like `crescendo`, `echo_chamber`, or `prompt_decomposition`) usually accept the `model=` parameter inside the `--attack-options`.

```bash
spikee test --dataset my_data.jsonl \
            --target always_refuse \
            --attack crescendo \
            --attack-options "model=openrouter-anthropic/claude-3.5-sonnet"
```

### Targets (via CLI `--target-options`)
Similarly, built-in API routing targets use `--target-options` to define the backend model they evaluate against.

```bash
spikee test --dataset my_data.jsonl \
            --target togetherai_api \
            --target-options "llama33-70b" \
            --attack llm_jailbreaker
```

*Note: For some built-in targets like `togetherai_api.py`, the code automatically prepends the prefix (`together-`) so the user only provides the raw litellm model string or internal shorthand.*

## End-to-End Example Command
Combine everything into a robust command that uses Groq for judging, OpenRouter for generating attacks, and TogetherAI as the application target:

```bash
spikee test --dataset evaluations.jsonl \
            --target togetherai_api \
            --target-options "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo" \
            --attack crescendo \
            --attack-options "model=openrouter-google/gemini-2.5-flash" \
            --judge-options "groq-llama-3.1-8b-instant"
```
