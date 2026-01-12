from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
from jinja2 import Template
model_name = "/model_path"
tokenizer = AutoTokenizer.from_pretrained(model_name)
import json
with open("/eval_data.json", 'r') as f:
    dataload = json.load(f)

data = []
for idx, item in tqdm(enumerate(dataload), total=len(dataload)):
    PAIR = []
    if "Pair-wise Comparative Feedback" in item and len(item["Pair-wise Comparative Feedback"]) != 0:
        PAIR = item["Pair-wise Comparative Feedback"]
    
    PAIR = PAIR[:4]

    comments = ""
    Like_Dislike = ""

    if len(PAIR) > 0:
        Like_Dislike = f"**This person has chosen or rejected comments on some posts:**\n\n"
        
        for i, it in enumerate(PAIR):
            Like_Dislike = (
                f"{Like_Dislike}"
                f"{i+1}. *Post:*\n{it['prompt']}\n\n"
                f"*Chosen:*\n{it['chosen']}\n\n"
                f"*Rejected:*\n{it['rejected']}\n\n"
            )

    dpo_prompt = (
        "Generate the user's preference based on their historical behavior.\n\n"
        f"{comments}{Like_Dislike}"
    )

    item["format"] = dpo_prompt
    item['idx'] = idx

    data.append(item)

sampling_params = SamplingParams(temperature=0.9, top_k=10, top_p=0.95, n=1, max_tokens=2048, 
                                stop=["User:", "Human:", "Assistant:", "<|im_end|>"],)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model=model_name,max_model_len=16384,tensor_parallel_size=4, disable_custom_all_reduce=True,)

texts = []
for i in range(len(data)):
    prop = data[i]["format"]
    #r1 distill
    messages = [
    {"role": "user", "content": "Please reason step by step.\n"+prop}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    texts.append(text)

# generate outputs
outputs = llm.generate(texts, sampling_params)

# Print the outputs.
lengs = []
tot = 0

for idx, output in enumerate(outputs):
    prompt = output.prompt
    data[idx]["pred_persona"] = output.outputs[0].text

import re

res = []
for item in data:
    r1_gen = item["pred_persona"]
    match = re.search(r'</think>(.*?)$', r1_gen, re.DOTALL)
    if match:
        result = match.group(1).strip()
    else:
        result = r1_gen

    item["profile"] = result
    res.append(item)

print(len(res))

with open("/pref_saved_path.json", 'w') as f:
    json.dump(res, f, indent=4)
