## Fetch & Convert the LMSYS Toxic Chat Dataset

This script downloads and converts the [LMSYS Toxic Chat](https://huggingface.co/datasets/lmsys/toxic-chat) dataset for use with Spikee.  
The dataset contains human-vetted toxic and jailbreaking prompts extracted from model conversations, making it ideal for evaluating safety guardrails.

### Prerequisites

- Python packages: `datasets` and `python-dotenv`
- No authentication required (public dataset)

### Fetch & Convert

Run the fetch-and-convert script from the root of your Spikee workspace:

```bash
python datasets/seeds-toxic-chat/fetch_and_convert_dataset.py
````

The script will:

* Download the `lmsys/toxic-chat` dataset (`toxicchat0124` split)
* Filter for **human-annotated** samples where `toxicity == 1` or `jailbreaking == 1`
* Convert them into Spikee’s JSONL format

After completion, the converted prompts will be available at:

```
datasets/seeds-toxic-chat/standalone_user_inputs.jsonl
```

### Generate the Spikee Test Dataset

Use the `spikee` CLI to generate a test dataset for Spikee’s tester script:

```bash
spikee generate \
  --seed-folder datasets/seeds-toxic-chat \
  --include-standalone-inputs
```

### Optional: Plugins & Variations

Apply additional transformations using plugins:

```bash
spikee generate \
  --seed-folder datasets/seeds-toxic-chat \
  --include-standalone-inputs \
  --plugins 1337 base64
```

### Dataset Information

* **Source**: `lmsys/toxic-chat`
* **Split**: `toxicchat0124`
* **Filtered Criteria**: `human_annotation == True` and (`toxicity == 1` or `jailbreaking == 1`)
* **Size (after filtering)**: 384 samples

  * 271 toxic samples
  * 113 toxic + jailbreak samples

* **Instruction Types**:

  * `lmsys-toxic-chat-toxic`
  * `lmsys-toxic-chat-toxic+jailbreak`
