# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the dataset to parquet format
V1: Hard to easy.
V2: Easy
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


def process_dialogue(history):
    prompt_str = "\n\n".join(
        [
            f"=====Tripe {idx+1}=====\n\n*QUERY:*\n {triple['query']}\n{'*'*5}\n*CHOSEN ITEM BY THE USER:*\n{triple['chosen']}\n{'*'*5}\n*REJECTED ITEM BY THE USER:*\n{triple['rejected']}"
            for idx, triple in enumerate(history)
        ]
    )

    full_text = f"""Analyze the past preference summary and the following user interaction history to summarize the user preferences in as concise language. If past preferences are provided, adjust the preferences by combining past preferences with those reflected in current behavior, removing conflicting parts, and integrating new insights. If no past preferences are provided, derive the final preferences solely from user behavior. The user's history will be provided as a sequence of triples, where each triple is `(QUERY, CHOSEN ITEM BY THE USER, REJECTED ITEM BY THE USER)`.\n=====Past Preference Summary=====\n\nNot Provided.\n\n=====Interaction History=====\n{prompt_str}\n\n=====END=====\n\nNow, given the above user's past prefernce summary and the interaction history, summarize the user preferences."""

    return full_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir", default="path for saving data"
    )
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    data_source_train = "path to jsonl data"

    train_dataset = datasets.load_dataset("json", data_files=data_source_train)
    train_dataset = train_dataset["train"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            mid_position = example["mid_position"]

            posts1 = example["history"][:mid_position]
            prompt1 = process_dialogue(posts1)
            target1 = example["history"][mid_position]
            posts2 = example["history"][mid_position:-1]
            prompt2 = process_dialogue(posts2)
            target2 = example["history"][-1]
            full_posts = example["history"][:-1]
            full_prompt = process_dialogue(full_posts)

            data = {
                "data_source": "upi",
                "prompt": [prompt1, prompt2, full_prompt],
                "ability": "personalization_alignment",
                "reward_model": {
                    "style": "rule",
                    "target1": target1,
                    "target2": target2,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "userid": example["uid"],
                    "prompt1": prompt1,
                    "prompt2": prompt2,
                    "full_prompt": full_prompt,
                    "mid_position": mid_position,
                    "history": example["history"],
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    train_dataset = train_dataset.remove_columns(
        [
            "uid",
            "history",
            "mid_position",
            "mid_record",
            "last_record",
            "signal_gap",
            "distance_to_end",
        ]
    )

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
