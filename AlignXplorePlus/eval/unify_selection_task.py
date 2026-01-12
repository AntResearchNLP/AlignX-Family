import json
import random

datasetname = "alignx"
input_filename = f"{datasetname}.json"
test_samples = []

with open(input_filename, "r") as f:
    samples = json.load(f)

for idx, item in enumerate(samples):
    his_list = item["Pair-wise Comparative Feedback"]
    history = []
    for his in his_list:
        chosen = his["chosen"]
        rejected = his["rejected"]
        query = his["prompt"]
        history.append({"query": query, "chosen": chosen, "rejected": rejected})
    target = {
        "query": item["prompt"],
        "chosen": item["chosen"],
        "rejected": item["rejected"],
    }
    test_samples.append(
        {
            "userid": f"{datasetname}_{idx}",
            "history": history,
            "target": target,
            "ground_truth_persona": item["profile_full"],
        }
    )

output_filename = f"alignx_upi.jsonl"
with open(output_filename, "w") as f:
    for item in test_samples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
