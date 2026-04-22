from datasets import load_dataset
import json
from dotenv import load_dotenv

# Load environment variables from .env (e.g., API keys)
load_dotenv()

# Load the wildguardmix dataset
dataset = load_dataset("allenai/wildguardmix", "wildguardtest")

# Filter for prompts where prompt_harm_label equals 'harmful'
not_harmful_prompts = dataset["test"].filter(
    lambda example: example["prompt_harm_label"] != "harmful"
)

# Create JSONL entries in the required format
jsonl_entries = []
for i, item in enumerate(not_harmful_prompts):
    entry = {
        "id": f"wildguard-{i + 1:02d}",
        "text": item["prompt"],
        "judge_name": "llm_judge_harmful",
        "judge_args": "",
        "instruction_type": item["subcategory"],
    }
    jsonl_entries.append(entry)

# Write to JSONL file
with open(
    "datasets/seeds-wildguardmix-harmful-fp/standalone_user_inputs.jsonl",
    "w",
    encoding="utf-8",
) as f:
    for entry in jsonl_entries:
        f.write(json.dumps(entry) + "\n")

print(f"Successfully converted {len(jsonl_entries)} prompts to JSONL format.")
print("Output saved to 'standalone_user_inputs.jsonl'")
