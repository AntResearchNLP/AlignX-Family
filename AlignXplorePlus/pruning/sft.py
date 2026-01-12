import sys
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk


# ========== 配置 ==========

MODEL_NAME = "Qwen/Qwen3-1.7B"
RAW_DATA_FILE = "sftdata.jsonl"
PROCESSED_DATA_DIR = "processed_dataset"

# ====== 检查数据是否已处理 ======
if not os.path.exists(PROCESSED_DATA_DIR):
    print(f"[ERROR] 未找到预处理好的数据: {PROCESSED_DATA_DIR}")
    print("请先运行 CPU 预处理脚本---preprocess.py")
    sys.exit(1)

print(f"[INFO] 从磁盘加载已处理的数据集: {PROCESSED_DATA_DIR} ...")
tokenized_dataset = load_from_disk(PROCESSED_DATA_DIR)
print(f"[INFO] 数据集加载完成，样本数: {len(tokenized_dataset)}")
print(tokenized_dataset.column_names)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

model.config.use_flash_attention = True

# ========== SFTTrainer 训练 ==========
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=1,
    num_train_epochs=1,
    learning_rate=1e-4,
    output_dir="./checkpoints",
    save_strategy="epoch",
    seed=42,
    bf16=True,
    report_to="tensorboard",
    dataloader_num_workers=64,
    dataloader_pin_memory=True,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
    args=training_args,
)

trainer.train()
