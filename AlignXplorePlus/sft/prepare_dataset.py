import os
from datasets import load_dataset
from transformers import AutoTokenizer


# 配置
model_name_or_path = "Qwen/Qwen3-8B"
data_file = "sft/sft.jsonl"
output_dir = "./tokenized_dataset"

def main():
    # os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    # 加载原始数据
    dataset = load_dataset("json", data_files=data_file, split="train")

    # 假设 JSONL 有字段 "text" 存放训练内容
    def tokenize_fn(example, processing_class, assistant_only_loss):
        output = {}
        prompt_ids = processing_class.apply_chat_template(
            example["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            tools=example.get("tools"),
            **example.get("chat_template_kwargs", {}),
        )
        # Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
        # even for single examples, while for LLMs it returns lists of ints.
        prompt_ids = prompt_ids[0] if isinstance(prompt_ids[0], list) else prompt_ids
        prompt_completion_processed = processing_class.apply_chat_template(
            example["prompt"] + example["completion"],
            return_dict=True,
            tokenize=True,
            return_assistant_tokens_mask=assistant_only_loss,
            tools=example.get("tools"),
            **example.get("chat_template_kwargs", {}),
        )
        # Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
        # even for single examples, while for LLMs it returns lists of ints.
        prompt_completion_processed = {
            k: v[0] if isinstance(v[0], list) else v
            for k, v in prompt_completion_processed.items()
        }
        prompt_completion_ids = prompt_completion_processed["input_ids"]
        if "assistant_masks" in prompt_completion_processed:
            output["assistant_masks"] = prompt_completion_processed["assistant_masks"]
        

        # Check if the tokenized prompt starts with the tokenized prompt+completion
        # print(prompt_completion_ids[: len(prompt_ids)])
        # print(prompt_ids)
        # print(prompt_completion_ids[: len(prompt_ids)] == prompt_ids)
        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
            print(
                "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                "token handling. Verify that the tokenizer is processing text consistently."
            )

        # Create completion mask
        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
        output["input_ids"] = prompt_completion_ids
        output["completion_mask"] = completion_mask

        if "assistant_masks" in output and 1 not in output["assistant_masks"]:
            raise RuntimeError(
                "You're using `assistant_only_loss=True`, but at least one example has no assistant "
                "tokens. This usually means the tokenizer's chat template doesn't generate assistant "
                "masks — it may be missing the `{% generation %}` keyword. Please check the template and "
                "ensure it's correctly configured to support assistant masking."
            )
        return output

    map_kwargs = {
        "num_proc":  128,
        # "batched": True
    }

    tokenized_dataset = dataset.map(
        tokenize_fn,
        fn_kwargs={
            "processing_class": tokenizer,
            "assistant_only_loss": False,
        },
        **map_kwargs,
    )

    # 保存成 arrow 格式
    tokenized_dataset.save_to_disk(output_dir)
    print(f"✅ Tokenized dataset saved to {output_dir}")

if __name__ == "__main__":
    main()
