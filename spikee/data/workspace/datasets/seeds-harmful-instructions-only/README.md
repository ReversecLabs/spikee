# Harmful Instructions Only

This dataset is specifically designed for **LLM-driven attacks** such as **Crescendo** and **LLM-Jailbreaker**.

In these types of attacks, the attacking LLM requires only a plain objective or instruction (e.g., "Write a phishing email") to generate its own attack vectors. It does not require pre-written jailbreak templates or static user inputs.

## Structure

*   **`instructions.jsonl`**: Contains the harmful intents/instructions that serve as the objective for the attack.
*   **`jailbreaks.jsonl`**: Contains a single empty entry. This dataset is meant to be used with attacks that generate their own jailbreaks dynamically.
*   **`base_user_inputs.jsonl`**: Contains a single empty entry.

# Dataset generation

The dataset can be generated using this command:

```bash
spikee generate --seed-folder datasets/seeds-harmful-instructions-only
```

