## Fetch & Convert the Simsonsun Jailbreak Prompts Dataset

This script downloads and converts the Simsonsun JailbreakPrompts dataset for use with Spikee. These datasets were specifically curated to avoid contamination with training data of popular jailbreak classifiers, enabling more accurate evaluation.

### Background

These datasets were constructed for the thesis: "Contamination Effects: How Training Data Leakage Affects Red Team Evaluation of LLM Jailbreak Detection". The study found that over 65% of red team evaluation prompts in popular tools (Garak, PyRIT, Giskard) were present in the training data of tested classifiers, leading to inflated performance metrics.

### Dataset Options

- **Dataset 1** (default): 67 manually selected, high-quality, structurally diverse jailbreak prompts
- **Dataset 2**: 2,359 broader coverage prompts to mitigate selection bias

### Prerequisites

- Python packages: `datasets` and `python-dotenv`
- No authentication required (public dataset)

### Fetch & Convert

Run the fetch-and-convert script from the root of your Spikee workspace:

```bash
# Default: Use Dataset 1 (high-quality filtered prompts)
python datasets/seeds-simsonsun-high-quality-jailbreaks/fetch_and_convert_dataset.py

# Optional: Use Dataset 2 (broader coverage)
python datasets/seeds-simsonsun-high-quality-jailbreaks/fetch_and_convert_dataset.py --dataset 2
```

After completion, the converted prompts will be available at:

```
datasets/seeds-simsonsun-high-quality-jailbreaks/standalone_user_inputs.jsonl
```

### Generate the Spikee Test Dataset

Use the `spikee` CLI to generate a test dataset:

```bash
spikee generate \
  --seed-folder datasets/seeds-simsonsun-high-quality-jailbreaks \
  --include-standalone-inputs
```

### Optional: Sample Testing

Since Dataset 2 is large (2,359 prompts), you might want to sample when testing:

```bash
# Test 10% of the dataset
spikee test --dataset datasets/simsonsun-high-quality-jailbreaks-full-prompt-dataset-1750077033.jsonl \
            --target your-target \
            --sample 0.1
```

### Dataset Information

- **Source**: Simsonsun/JailbreakPrompts
- **Purpose**: Contamination-free evaluation of LLM guardrails
- **Dataset 1**: 67 high-quality, diverse prompts
- **Dataset 2**: 2,359 broader coverage prompts
