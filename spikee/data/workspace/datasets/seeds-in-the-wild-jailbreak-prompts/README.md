## Fetch & Convert the In-The-Wild Jailbreak Prompts Dataset

This script downloads and converts the TrustAIRLab in-the-wild jailbreak prompts dataset for use with Spikee. The dataset contains real jailbreak attempts collected from various platforms.

### Prerequisites

- Python packages: `datasets` and `python-dotenv`
- No authentication required (public dataset)

### Fetch & Convert

Run the fetch-and-convert script from the root of your Spikee workspace:

```bash
python datasets/seeds-in-the-wild-jailbreak-prompts/fetch_and_convert_dataset.py
```

The script will download the dataset and filter for actual jailbreak prompts (where `jailbreak=True`).

After completion, the converted prompts will be available at:

```
datasets/seeds-in-the-wild-jailbreak-prompts/standalone_user_inputs.jsonl
```

### Generate the Spikee Test Dataset

Use the `spikee` CLI to generate a test dataset for Spikee's tester script:

```bash
spikee generate \
  --seed-folder datasets/seeds-in-the-wild-jailbreak-prompts \
  --include-standalone-inputs
```

### Optional: Plugins & Variations

Apply additional transformations using plugins:

```bash
spikee generate \
  --seed-folder datasets/seeds-in-the-wild-jailbreak-prompts \
  --include-standalone-inputs \
  --plugins 1337 base64
```

### Dataset Information

- **Source**: TrustAIRLab/in-the-wild-jailbreak-prompts
- **Collection Date**: 2023-12-25
- **Platforms**: Various (Discord, Reddit, etc.)
- **Size**: ~1,400 prompts (filtered to actual jailbreaks)

