from datasets import load_dataset
import json
import os
from dotenv import load_dotenv

load_dotenv()

print("Loading lmsys/toxic-chat...")
dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
train = dataset["train"]

print("Filtering human-vetted toxic/jailbreaking samples...")
filtered = [
    x
    for x in train
    if x["human_annotation"] and (x["toxicity"] == 1 or x["jailbreaking"] == 1)
]

jsonl_entries = []
for i, item in enumerate(filtered):
    if item["toxicity"] == 1 and item["jailbreaking"] == 1:
        instruction_type = "lmsys-toxic-chat-toxic+jailbreak"
    elif item["toxicity"] == 1:
        instruction_type = "lmsys-toxic-chat-toxic"
    elif item["jailbreaking"] == 1:
        instruction_type = "lmsys-toxic-chat-jailbreak"
    else:
        instruction_type = "lmsys-toxic-chat-other"

    entry = {
        "id": f"toxicchat-{i + 1:04d}",
        "text": item["user_input"],
        "judge_name": "llm_judge_harmful",
        "judge_args": "",
        "instruction_type": instruction_type,
    }
    jsonl_entries.append(entry)

os.makedirs("datasets/seeds-toxic-chat", exist_ok=True)
output_path = "datasets/seeds-toxic-chat/standalone_user_inputs.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for entry in jsonl_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"\nSuccessfully converted {len(jsonl_entries)} toxic/jailbreaking prompts.")
print(f"Output saved to '{output_path}'")

# optional summary
count = {}
for e in jsonl_entries:
    k = e["instruction_type"]
    count[k] = count.get(k, 0) + 1
print("\nPrompts by instruction_type:")
for k, v in sorted(count.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k}: {v}")
