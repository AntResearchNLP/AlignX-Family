import os

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator


def main():
    # accelerator = Accelerator()

    # 1. 获取分布式世界大小与 rank 信息
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"Distributed training: local_rank={local_rank}, world_size={world_size}")

    # os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # 2. 加载模型和分词器
    model_name_or_path = "Qwen/Qwen3-8B"  # 替换为您想微调的模型
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.bfloat16,  # 使用bfloat16以节省显存
        # device_map={"": accelerator.device},
        trust_remote_code=True,
    )
    model.config.use_flash_attention = True

    # 3. 加载并准备数据集
    dataset = load_from_disk("tokenized_dataset")

    # 4. 初始化训练配置和SFTTrainer
    sft_config = SFTConfig(
        dataset_num_proc=128,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        logging_steps=100,
        num_train_epochs=4,
        learning_rate=2e-5,  # 5e-5/1e-4?
        output_dir="./checkpoints",
        save_strategy="epoch",
        eval_strategy="no",
        seed=42,
        bf16=True,
        report_to="tensorboard",
        save_total_limit=1,
        dataloader_num_workers=64,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        max_length=16384,
        # deepspeed="./ds_config.json",
        warmup_ratio=0.1,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        tf32=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
        # SFTTrainer会自动使用dataset_text_field指定的列，无需再传formatting_func
        # 如果数据集未预处理，则可在此处传入formatting_func
    )

    # 5. 开始训练
    print("开始模型微调...")
    trainer.train(resume_from_checkpoint=True)

if __name__ == "__main__":
    main()
