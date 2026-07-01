# LLM Providers in Spikee

A Provider Module is a Python script that abstracts interactions (e.g., authentication, invocation, formatting) with a specific LLM provider. While Spikee contains several built-in modules, leveraging the `any-llm` library for popular LLM providers, users can create their own custom provider modules to interface with any LLM service they choose. These providers can be accessed by a wide range of other modules (e.g., Targets, Attacks, Plugins, Judges) to utlise LLM technologies.

## Installation and Dependencies
Spikee uses `any-llm` to handle interactions with providers. To keep Spikee as lightweight as possible, **only dependencies for OpenAI-compatible API endpoints** are installed by default (`pip install spikee`). This base installation supports OpenAI, DeepSeek, OpenRouter, Google, TogetherAI, and Custom providers out of the box.

If you want to use targets, judges, or attackers powered by providers for which spikee relies on native SDKs (such as AWS Bedrock, Azure, Ollama or Groq), you explicitly need to install those optional dependencies:
```bash
pip install "spikee[all]"
# Or specifically: "spikee[bedrock,azure,ollama,groq]"
```

## Identifying Providers and Models

All providers and models can be specified using the following format:

- Full Identifier: `<provider-prefix>/<model-name>`
    - Examples:
        - `openai/gpt-4o`,
        - `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`,
        - `bedrock/claude45-sonnet`, (some providers support shorthand model names)
        - `google/gemini-3.0-pro`


- Provider Identifier (uses provider's default model): `<provider-prefix>/`
    - Examples:
        - `openai/` (defaults to `openai/gpt-4o`),
        - `bedrock/` (defaults to `bedrock/claude45-sonnet`),
        - `google/` (defaults to `google/gemini-2.5-flash`)

Use `spikee list providers` to get a list of providers and known supported models / shorthands.

## Built-In Providers

| LLM Provider | Provider ID | Default / Example Models | Environment Variables | External Docs |
| :--- | :--- | :--- | :--- | :--- |
| OpenAI | `openai` | `gpt-4o` (default)<br/>`gpt-4.1` | `OPENAI_API_KEY` | [Models List](https://platform.openai.com/docs/models) |
| Azure OpenAI | `azure` | `gpt-4o` (default)<br/>`gpt-4o-mini` | `AZURE_OPENAI_API_KEY`<br/>`AZURE_OPENAI_ENDPOINT` | [Models List](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models) |
| AWS Bedrock | `bedrock` | `claude45-sonnet` (default)<br/>`claude45-haiku`<br/>`deepseek-v3`<br/><small>*(Allows internal shorthands)*</small> | `AWS_ACCESS_KEY_ID`<br/>`AWS_SECRET_ACCESS_KEY`<br/>`AWS_DEFAULT_REGION` | [Models List](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) |
| Google Gemini | `google` | `gemini-2.5-flash` (default)<br/>`gemini-2.5-pro`<br/>`gemini-3-pro` | `GOOGLE_API_KEY` | [Models List](https://ai.google.dev/gemini-api/docs/models/gemini) |
| Deepseek | `deepseek` | `deepseek-chat` (default)<br/>`deepseek-reasoner` | `DEEPSEEK_API_KEY` | [Models List](https://platform.deepseek.com/api-docs/) |
| Groq | `groq` | `llama-3.1-8b-instant` (default)<br/>`llama-3.3-70b-versatile` | `GROQ_API_KEY` | [Models List](https://console.groq.com/docs/models) |
| TogetherAI | `together` | `gemma2-8b` (default)<br/>`mixtral-8x22b`<br/><small>*(Allows internal shorthands)*</small> | `TOGETHER_API_KEY` | [Models List](https://docs.together.ai/docs/inference-models) |
| OpenRouter | `openrouter` | `google/gemini-2.5-flash` (default)<br/>`anthropic/claude-3.5-haiku` | `OPENROUTER_API_KEY` | [Models List](https://openrouter.ai/models) |
| Local (Ollama) | `ollama` | *None* | `OLLAMA_URL` | |
| Local (LLaMA CPP Server) | `llamacpp` | *None* | `LLAMACPP_URL` | |
| Custom | `custom` | *None* | `CUSTOM_API_URL`<br/>`CUSTOM_API_KEY` | *Custom OpenAI-Based API* |
| Offline | `offline` | `offline` | *None* | [See Judges section](./09_judges.md#1-scan-using-offline-judge) |


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
# Full Identifier
spikee test --dataset my_data.jsonl \
            --target llm_provider \
            --target-options "bedrock/claude45-haiku" \
            --judge-options "openai/gpt-4o-mini"

# Provider Identifier (defaults to the provider's default model)
# Full Identifier
spikee test --dataset my_data.jsonl \
            --target llm_provider \
            --target-options "bedrock/claude45-haiku" \
            --judge-options "openai/" # Defaults to "openai/gpt-4o"
```

### Attacks (via CLI `--attack-options`) and Plugins (via CLI `--plugin-options`)
Attacks and plugins that leverage LLMs dynamically to generate payloads (like `crescendo`, `echo_chamber`, or `llm_jailbreaker`) usually accept the `model=` parameter inside the `--attack-options` or `--plugin-options` arguments.

```bash
spikee test --dataset my_data.jsonl \
            --target llm_provider \
            --target-options "deepseek/deepseek-chat" \
            --attack crescendo \
            --attack-options "model=bedrock/claude45-sonnet"
```

### Targets (via CLI `--target-options`)
Similarly, built-in API routing targets use `--target-options` to define the backend model they evaluate against.

```bash
spikee test --dataset my_data.jsonl \
            --target llm_provider \
            --target-options "google/gemini-2.5-flash" \
            --attack llm_jailbreaker
```


### End-to-End Example
Combine everything into a robust command that uses Groq for judging, OpenRouter for generating attacks, and TogetherAI as the application target:

```bash
spikee test --dataset evaluations.jsonl \
            --target llm_provider \
            --target-options "google/gemini-2.5-flash" \
            --attack crescendo \
            --attack-options "model=openrouter/google/gemini-2.5-flash" \
            --judge-options "groq/llama-3.1-8b-instant"
```

## Global Timeouts

When using LLMs for modules like judging and dynamic attacks, you might occasionally need to increase the timeout for the underlying requests. This is especially true if you are using complex multi-turn attacks or running local servers (`llama.cpp`, `ollama` etc) that don't have powerful GPUs and need more time to process large contexts.

You can override the default API timeout (typically 600 seconds) across all Spikee LLM providers by setting the `SPIKEE_API_TIMEOUT` environment variable (in seconds) before running your testing harness.

```bash
# E.g. Set a 20-minute global timeout across all LLM providers
SPIKEE_API_TIMEOUT=1200.0 spikee test --dataset my_dataset.jsonl --target llm_provider
```

## Debugging Provider Calls

Setting `PROVIDER_DEBUG=true` in your environment (or `.env` file) enables verbose prompt and response logging for every provider call. Each message sent to and received from the LLM will be printed to the console, prefixed with the provider class name.

This is useful for verifying the exact messages being sent to the LLM, especially when debugging custom targets, attacks, or judges that use `get_llm()` internally.

## Implementing Built-In LLM Utilities
Spikee's built-in LLM utility is implemented within `Provider` modules, and can be obtained using the `get_llm()` function from `spikee/utilities/llm.py`.


The following example demonstrates how to obtain and invoke an LLM:
```python
from spikee.utilities.llm import get_llm
from spikee.utilities.llm_message import SystemMessage, HumanMessage

model = "bedrock/claude37-sonnet"

llm = get_llm(model, max_tokens=2048, temperature=0.7)

response = llm.invoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]).content

print(response)  # Should print "Paris"
```

## Creating Custom Provider Modules
Custom provider modules are located within the `providers/` directory of your Spikee workspace, and identify themselves by their filename (e.g., `bedrock.py` has the provider prefix `bedrock/`).

> **Note:** `invoke()` is a legacy implementation and should not be overridden in subclasses. Implement **`_invoke()`** instead — it is called internally by `invoke()`, which handles debug logging and error wrapping.

### `ProviderError`

When an unrecoverable error occurs inside `_invoke()`, raise a `ProviderError` rather than a generic exception. This gives callers structured access to the prompt, response, and error metadata.

```python
from spikee.templates.provider import ProviderError

# Raise with full context
raise ProviderError(
    "Received empty response due to max_tokens budget exceeded during thinking.",
    prompt=messages,      # MessageHint — the input that caused the error
    response=response,    # AIMessage | None — the (partial) response if available
    metadata={"finish_reason": "length"},  # any extra diagnostic data
)
```

Exceptions that are not already `ProviderError` instances are automatically caught by `invoke()` and re-raised as `ProviderError`, so you only need to raise it explicitly when you want to attach context (e.g., provider-specific validation failures).

### Provider Template

The following template demonstrates how to create a custom provider module by inheriting from the `Provider` base class:

```python
from spikee.templates.provider import Provider, ProviderError
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import format_messages, AIMessage, MessageHint
from spikee.utilities.hinting import ModuleDescriptionHint

from typing import Dict, Union, Any


class ExampleProvider(Provider):

    @property
    def default_model(self) -> str:
        # If default_model is not defined, the first model in the models dict will be used as the default
        return "mock"

    @property
    def models(self) -> Dict[str, str]:
        return {
            "mock": "mock",
        }

    def setup(self, model: str, max_tokens: Union[int, None] = None, temperature: Union[float, None] = None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Map user-friendly model names to model identifiers
        self.model = self.models.get(self.model, self.model)

        self.llm = ...

    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.LLM], "Sample LLM Provider, always returns 'Hello, world!'"

    def _invoke(self, messages: MessageHint) -> AIMessage:
        """Return 'Hello, world!' regardless of the input."""

        # Convert messages into a standard dict format
        messages = format_messages(messages)

        response = ...  # Your logic to call the LLM API with the formatted messages

        return AIMessage(content="Hello, world!", original_response=...)
```

### Inheriting from `AnyLLMCustomProvider`

LLM Providers utilising an OpenAI-compatible API can also inherit from `AnyLLMCustomProvider` — this is an example for a custom Google Gemini provider:

```python
from spikee.providers.custom import AnyLLMCustomProvider
from spikee.templates.provider import ProviderError
from spikee.utilities.llm_message import AIMessage, MessageHint
from typing import Dict, Union
import os

class AnyLLMGoogleProvider(AnyLLMCustomProvider):
    """AnyLLM provider for Google models (via Custom provider with OpenAI compatibility)"""

    @property
    def default_model(self) -> str:
        return "gemini-2.5-flash"

    @property
    def models(self) -> Dict[str, str]:
        return {
            # Gemini 3 (Latest)
            "gemini-3.1-pro": "gemini-3.1-pro",
            "gemini-3.1-flash": "gemini-3.1-flash",

            # Gemini 2.5
            "gemini-2.5-pro": "gemini-2.5-pro",
            "gemini-2.5-flash": "gemini-2.5-flash",
        }

    @property
    def name(self) -> str:
        return "Google"

    @property
    def base_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta/openai/"

    @property
    def api_key(self) -> Union[str, None]:
        return os.getenv("GOOGLE_API_KEY", None)
```

#### `response_validation()` Hook

Providers that inherit from `AnyLLMCustomProvider` (the OpenAI-compatible base) can override `response_validation()` to add provider-specific checks after a response is received. The default implementation is a no-op.

```python
def response_validation(self, messages: MessageHint, response: AIMessage) -> None:
    """Called by _invoke() after a successful API call. Raise ProviderError on invalid responses."""
    pass
```