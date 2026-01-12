set -x


# 定义数据集列表
rec_datasets=(
    "amazon"
    "mind"
    "movielens"
)


select_datasets=(
    "alignx"
    "concise_detail"
    "friendly_unfriendly"
    "student_phd"
    "personamem"
    # "PRISM"
)

model="Qwen3_8B"

# 循环执行
for dataset in "${rec_datasets[@]}"; do
    echo "Evaluating $dataset ..."
    python -u evaluate_rec_pair.py --input_file=preferences/${model}/${dataset}_upi_preference_${model}.json --model_name=Qwen/Qwen3-8B
done

for dataset in "${select_datasets[@]}"; do
    echo "Evaluating $dataset ..."
    python -u evaluate_select.py --input_file=preferences/${model}/${dataset}_upi_preference_${model}.json --model_name=Qwen/Qwen3-8B
done

# 循环执行
for dataset in "${rec_datasets[@]}"; do
    echo "Evaluating $dataset ..."
    python -u evaluate_rec_pair.py --input_file=preferences/${model}/${dataset}_upi_preference_streaming_${model}.json --model_name=Qwen/Qwen3-8B
done

for dataset in "${select_datasets[@]}"; do
    echo "Evaluating $dataset ..."
    python -u evaluate_select.py --input_file=preferences/${model}/${dataset}_upi_preference_streaming_${model}.json --model_name=Qwen/Qwen3-8B
done

echo "Done"