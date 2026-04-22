from datasets import load_dataset
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Load the in-the-wild jailbreak prompts dataset
print("Loading dataset from Hugging Face...")
dataset = load_dataset(
    "TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25"
)

# Filter for actual jailbreak prompts (jailbreak=True)
print("Filtering for jailbreak prompts...")
jailbreak_prompts = [item for item in dataset["train"] if item["jailbreak"]]

# Create JSONL entries in the required format
jsonl_entries = []
for i, item in enumerate(jailbreak_prompts):
    entry = {
        "id": f"inthewild-{i + 1:04d}",
        "text": item["prompt"],
        "judge_name": "llm_judge_harmful",
        "judge_args": "",
        "instruction_type": f"in-the-wild-jailbreak-prompts-{item.get('platform', 'unknown')}",
    }
    jsonl_entries.append(entry)

# Ensure output directory exists
os.makedirs("datasets/seeds-in-the-wild-jailbreak-prompts", exist_ok=True)

# Write to JSONL file
output_path = (
    "datasets/seeds-in-the-wild-jailbreak-prompts/standalone_user_inputs.jsonl"
)
with open(output_path, "w", encoding="utf-8") as f:
    for entry in jsonl_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(
    f"\nSuccessfully converted {len(jsonl_entries)} jailbreak prompts to JSONL format."
)
print(f"Output saved to '{output_path}'")

# Print some statistics
platforms = {}
for item in jailbreak_prompts:
    platform = item.get("platform", "unknown")
    platforms[platform] = platforms.get(platform, 0) + 1

print("\nPrompts by platform:")
for platform, count in sorted(platforms.items(), key=lambda x: x[1], reverse=True):
    print(f"  {platform}: {count}")
