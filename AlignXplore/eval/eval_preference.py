from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
from jinja2 import Template
model_name = "/Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
import json
with open("/eval_data.json", 'r') as f:
    dataload = json.load(f)

with open('/prefrence.txt', 'r') as f:
    system_prompt = f.read()

with open("/pref_saved_path.json", 'r') as f:
    data_B = json.load(f)

prompt_to_b_data = {}
for item in data_B:
    prompt = item.get('prompt')
    if prompt:
        prompt_to_b_data[prompt] = item

data = []
for idx, item in tqdm(enumerate(dataload), total=len(dataload)):
    prompt = item["prompt"]
    responseA = item["responseA"]
    responseB = item["responseB"]
    # persona = item["gt_persona"]

    if prompt in prompt_to_b_data:
        corresponding_b_data = prompt_to_b_data[prompt]
        persona = corresponding_b_data["profile"]
    else:
        persona = ""

    # input_format = (f'Determine which response the user prefers. Please output your selection below in a json format by filling in the placeholders in []:{{"selection": "[Response A / Response B]"}}\n{system_prompt}\n<Prompt>\n{prompt}\n</Prompt>\n\n<Response A>\n{responseA}\n</Response A>\n\n<Response B>\n{responseB}\n</Response B>\n\n')
    
    input_format = (f'Determine which response the user prefers based on the user’s preferences. Please output your selection below in a json format by filling in the placeholders in []:{{"selection": "[Response A / Response B]"}}\n{system_prompt}\n<Prompt>\n{prompt}\n</Prompt>\n\n<Preference>\n{persona}</Preference>\n\n<Response A>\n{responseA}\n</Response A>\n\n<Response B>\n{responseB}\n</Response B>\n\n')

    item["format"] = input_format
    item['idx'] = idx

    data.append(item)

sampling_params = SamplingParams(temperature=1, n=1, max_tokens=128, 
                                stop=["User:", "Human:", "Assistant:", "<|im_end|>"],)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model=model_name,max_model_len=16384,tensor_parallel_size=4, disable_custom_all_reduce=True,)

texts = []
for i in range(len(data)):
    prop = data[i]["format"]

    ##qwen2.5-7B-ins
    messages = [
        {"role": "system", "content": "Generate a task-specific response."},
        {"role": "user", "content": prop}
    ]

    # ##Qwq32B
    # messages = [
    # {"role": "user", "content": "Generate a task-specific response.\nPlease reason step by step.\n"+prop}
    # ]

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

import re
for idx, output in enumerate(outputs):
    prompt = output.outputs[0].text
    # breakpoint()

    ##qwen2.5-7B-ins
    try:
        prompt = json.loads(prompt)
    except:
        try:
            match = re.search(r'\{.*\}', prompt)
            if match:
                json_str = match.group()
                parsed_json = json.loads(json_str)
            prompt = parsed_json
        except:
            print(prompt)
            continue

    try:
        data[idx]["selection"] = prompt["selection"]
    except:
        data[idx]["selection"] = prompt

    # ##Qwq32B
    # match = re.search(r'</think>(.*?)$', prompt, re.DOTALL)
    # if match:
    #     prompt = match.group(1).strip()
    # else:
    #     prompt = prompt
    # prompt = prompt.strip()

    # try:
    #     prompt = json.loads(prompt)
    # except:
    #     try:
    #         match = re.search(r'\{.*\}', prompt)
    #         if match:
    #             json_str = match.group()
    #             parsed_json = json.loads(json_str)
    #         prompt = parsed_json
    #     except:
    #         print(prompt)
    #         continue

    # try:
    #     data[idx]["selection"] = prompt["selection"]
    # except:
    #     data[idx]["selection"] = prompt

with open("/ans_path.json", 'w') as f:
    json.dump(data, f, indent=4)

total = len(data)
count = 0
for item in data:
    if "selection" in item:
        if item["selection"] == item["answer"]:
            count += 1

print("acc:", count/total)
print(count)
print(total)