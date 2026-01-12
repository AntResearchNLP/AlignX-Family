import json
import os
import math
from pathlib import Path
from typing import List, Dict
import multiprocessing as mp
from collections import defaultdict


def load_jsonl(path: str) -> List[Dict]:
    """读取 JSONL 文件"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def select_records(records: List[Dict], a: float, b: float, c: float) -> List[Dict]:
    """
    筛选逻辑：
    1. 按 signal 降序取前 a%
    2. 过滤 logprob1 >= log(b)
    3. 按 logprob1 升序/降序取前 c%;
    """
    sorted_by_signal = sorted(records, key=lambda x: x["signal"], reverse=True)
    n1 = max(1, int(len(sorted_by_signal) * a))
    top_a = sorted_by_signal[:n1]

    log_b = math.log(b)
    log_upper = math.log(1.0)
    filtered_by_logprob = [
        r for r in top_a if r["logprob1"] >= log_b and r["logprob1"] <= log_upper
    ]

    # Version2:: 降序取前c%
    sorted_by_logprob = sorted(
        filtered_by_logprob, key=lambda x: x["logprob1"], reverse=True
    )
    c1 = max(1, int(len(sorted_by_logprob) * (c - 0.1)))
    c2 = max(1, int(len(sorted_by_logprob) * c))
    final_subset = sorted_by_logprob[c1:c2]

    return final_subset


def group_by_uid(records: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for r in records:
        grouped[r["uid"]].append(r)
    return grouped


def select_user_targets(
    uid: str, candidate_records: List[Dict], all_records: List[Dict]
) -> Dict:
    """
    从候选记录选出两条目标记录，满足：
      - 长 history 长度 >= 4
      - mid_position != 0
      - mid_position != len(long_history) - 2
    如果找不到符合要求的组合，返回 None
    """
    # 全历史按 pref_idx 升序
    all_sorted = sorted(all_records, key=lambda x: x["pref_idx"])
    candidates_sorted = sorted(
        candidate_records, key=lambda x: x["signal"], reverse=True
    )

    # 判断长度剪枝
    longest_history_len = 0
    for rec in candidates_sorted:
        target_idx = rec["pref_idx"]
        hist_len = len([r for r in all_sorted if r["pref_idx"] <= target_idx])
        longest_history_len = max(longest_history_len, hist_len)
    if longest_history_len < 4:
        return None

    # 遍历候选组合
    for i in range(len(candidates_sorted)):
        for j in range(i + 1, len(candidates_sorted)):
            r1 = candidates_sorted[i]
            r2 = candidates_sorted[j]
            # 谁在历史中更晚就是长记录
            if r1["pref_idx"] > r2["pref_idx"]:
                long_rec, short_rec = r1, r2
            else:
                long_rec, short_rec = r2, r1

            long_hist_items = [
                r["original_history_item"]
                for r in all_sorted
                if r["pref_idx"] <= long_rec["pref_idx"]
            ]
            short_hist_items = [
                r["original_history_item"]
                for r in all_sorted
                if r["pref_idx"] <= short_rec["pref_idx"]
            ]

            if short_hist_items[-1] not in long_hist_items:
                continue

            mid_position = long_hist_items.index(short_hist_items[-1])

            # 条件检查
            if mid_position != 0 and mid_position != len(long_hist_items) - 2:
                mid_record_info = next(
                    (
                        {
                            "logprob1": r["logprob1"],
                            "logprob2": r["logprob2"],
                            "signal": r["signal"],
                        }
                        for r in all_sorted
                        if r["pref_idx"] == short_rec["pref_idx"]
                    ),
                    None,
                )
                last_record_info = next(
                    (
                        {
                            "logprob1": r["logprob1"],
                            "logprob2": r["logprob2"],
                            "signal": r["signal"],
                        }
                        for r in all_sorted
                        if r["pref_idx"] == long_rec["pref_idx"]
                    ),
                    None,
                )
                # 额外统计字段
                signal_gap = (last_record_info["signal"] if last_record_info else 0) - (
                    mid_record_info["signal"] if mid_record_info else 0
                )
                distance_to_end = len(long_hist_items) - 1 - mid_position

                return {
                    "uid": str(uid),
                    "history": long_hist_items,
                    "mid_position": mid_position,
                    "mid_record": mid_record_info,
                    "last_record": last_record_info,
                    "signal_gap": signal_gap,
                    "distance_to_end": distance_to_end,
                }
    return None


def process_split_file(path: str, output_dir: str, a: float, b: float, c: float):
    print(f"[INFO] Processing {path}")
    all_data = load_jsonl(path)

    # 构建全量用户映射
    all_by_uid = group_by_uid(all_data)

    # 全局 a/b/c 筛选
    subset = select_records(all_data, a, b, c)
    subset_by_uid = group_by_uid(subset)

    outputs = []
    total_len = 0
    count_users = 0

    for uid, recs in subset_by_uid.items():
        res = select_user_targets(uid, recs, all_by_uid[uid])
        if res:
            outputs.append(res)
            total_len += len(res["history"])
            count_users += 1

    avg_len = total_len / count_users if count_users else 0
    print(f"[STAT] {path}: users={count_users}, avg_long_history_len={avg_len:.2f}")

    out_path = Path(output_dir) / (Path(path).stem + "_test_easy.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[INFO] Finished {path}, results: {len(outputs)}, saved to {out_path}")


def process_all_splits(
    input_dir: str, output_dir: str, a: float, b: float, c: float, num_workers: int = 4
):
    os.makedirs(output_dir, exist_ok=True)
    files = list(Path(input_dir).glob("*_once.jsonl"))
    with mp.Pool(processes=num_workers) as pool:
        args = [(str(f), output_dir, a, b, c) for f in files]
        pool.starmap(process_split_file, args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input dir")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dir")
    parser.add_argument("--a", type=float, required=True, help="Signal top a% (0~1)")
    parser.add_argument("--b", type=float, required=True, help="logprob1 >= log(b)")
    parser.add_argument(
        "--c",
        type=float,
        required=True,
        help="Within filtered, take bottom c% of logprob1",
    )
    parser.add_argument("--workers", type=int, default=10, help="并行进程数")
    args = parser.parse_args()

    process_all_splits(
        args.input_dir, args.output_dir, args.a, args.b, args.c, args.workers
    )
