This example dataset can be used for testing topical guardrails aimed at blocking prompts/queries/instructions for an LLM system that attempt to elicit "personal investment/financial advice" queries.

### Usage

If you are testing a **chatbot-style application**, use `--format user-input`.  
In this mode, the dataset is built from `base_user_inputs.jsonl`, i.e. common user questions for that chatbot:

```bash
spikee generate --seed-folder datasets/seeds-investment-advice --include-standalone-inputs --format user-input
```

If you are testing an **LLM in isolation**, use `--format full-prompt`.
Here you can also include a system prompt (via `--include-system-message`) to reflect the model’s deployed configuration:

```bash
spikee generate --seed-folder datasets/seeds-investment-advice --include-standalone-inputs --format full-prompt --include-system-message
```
