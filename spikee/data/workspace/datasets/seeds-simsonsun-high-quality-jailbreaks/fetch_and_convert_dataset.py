from datasets import load_dataset
import json
import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Convert Simsonsun JailbreakPrompts dataset to Spikee format"
)
parser.add_argument(
    "--dataset",
    type=int,
    choices=[1, 2],
    default=1,
    help="Which dataset to use: 1 (high-quality, default) or 2 (broad coverage)",
)
args = parser.parse_args()

# Load the Simsonsun JailbreakPrompts dataset
print("Loading dataset from Hugging Face...")
dataset_dict = load_dataset("Simsonsun/JailbreakPrompts")

# Select the appropriate dataset
if args.dataset == 1:
    dataset = dataset_dict["Dataset_1"]
    dataset_prefix = "simsonsun-ds1"
    print("Using Dataset 1: High-quality filtered prompts")
else:
    dataset = dataset_dict["Dataset_2"]
    dataset_prefix = "simsonsun-ds2"
    print("Using Dataset 2: Broad coverage prompts")

# Create JSONL entries in the required format
jsonl_entries = []
for i, item in enumerate(dataset):
    source = item["Source"].lower().replace(" ", "-").replace("_", "-")
    entry = {
        "id": f"{dataset_prefix}-{i + 1:04d}",
        "text": item["Prompt"],
        "judge_name": "llm_judge_harmful",
        "judge_args": "",
        "instruction_type": f"{dataset_prefix}-{source}",
    }
    jsonl_entries.append(entry)

# Ensure output directory exists
os.makedirs("datasets/seeds-simsonsun-high-quality-jailbreaks", exist_ok=True)

# Write to JSONL file
output_path = (
    "datasets/seeds-simsonsun-high-quality-jailbreaks/standalone_user_inputs.jsonl"
)
with open(output_path, "w", encoding="utf-8") as f:
    for entry in jsonl_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(
    f"\nSuccessfully converted {len(jsonl_entries)} jailbreak prompts to JSONL format."
)
print(f"Output saved to '{output_path}'")

# Print source statistics
sources = {}
for item in dataset:
    source = item["Source"]
    sources[source] = sources.get(source, 0) + 1

print("\nPrompts by source:")
for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
    print(f"  {source}: {count}")
