from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
from jinja2 import Template
model_name = "/the model you have trained"
tokenizer = AutoTokenizer.from_pretrained(model_name)
import json
with open("/eval_data.json", 'r') as f:
    dataload = json.load(f)

def process_dialogue(input_prompt):
        prompt_template_jinja = """\
{{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
Assistant: <think>\
"""

        prompt_instruction_template_jinja = """\
You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>.
This is the problem:
{{prompt}}
"""

        prompt_instruction_template = Template(prompt_instruction_template_jinja)
        prompt_instruction = prompt_instruction_template.render(prompt=input_prompt)
        prompt_template = Template(prompt_template_jinja)
        if tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = tokenizer.decode([tokenizer.bos_token_id])
        prompt = prompt_template.render(bos_token=bos_token, prompt=prompt_instruction)

        return prompt

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

    item["format"] = process_dialogue(dpo_prompt)
    item['idx'] = idx

    data.append(item)

sampling_params = SamplingParams(temperature=0.9, top_k=10, top_p=0.95, n=1, max_tokens=2048, 
                                stop=["User:", "Human:", "Assistant:", "<|im_end|>"],)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model=model_name,max_model_len=16384,tensor_parallel_size=4, disable_custom_all_reduce=True,)

texts = []
for i in range(len(data)):
    prop = data[i]["format"] 
    texts.append(prop)

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
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = re.findall(pattern, r1_gen)
    if matches:
        result = matches[-1].replace("answer here", "")
    else:
        result = r1_gen

    item["profile"] = result
    res.append(item)

print(len(res))

with open("/pref_saved_path.json", 'w') as f:
    json.dump(res, f, indent=4)
