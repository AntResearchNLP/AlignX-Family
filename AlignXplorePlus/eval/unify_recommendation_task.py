import json
import random

"""
#Question:
Given the user’s historical interactions and preferences, please determine whether the user will enjoy the target new movie by answering "Yes" or "No". 
User’s liked items: {}.
User’s disliked items: {}.
User's Preferences: {}.
Target new movie: {}
#Answer:
"""


random.seed(42)
dataset = "~/data/MIND/test.jsonl"
test_samples = []
with open(dataset, "r") as f:
    for line in f:
        item = json.loads(line)
        history = item["behaviors"]
        for his in history:
            if his["query"] is None:
                his["query"] = "Please recommend some news to me."
        last = history[-1]
        test_samples.append(
            {
                "userid": str(item["uid"]),
                "history": history[:-1],
                "target": last,
            }
        )

final_samples = random.sample(test_samples, 3000)
output_filename = "benchmarks/mind_upi.jsonl"
with open(output_filename, "w") as f:
    for item in final_samples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
