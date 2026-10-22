"""Download the real datasets and cut the subsamples this project trains on.

Datasets (both public, from Hugging Face):
  - tatsu-lab/alpaca: 52,002 instruction examples. License CC BY-NC 4.0.
    Used for SFT and for the held-out eval set.
  - HuggingFaceH4/ultrafeedback_binarized (split train_prefs): human and
    model preference pairs from UltraFeedback. Used for DPO, converted
    here to plain prompt/chosen/rejected strings.

The subsample sizes and the seed are fixed so anyone can regenerate the
exact same files:

    python scripts/prepare_data.py

Outputs (gitignored, regenerated on demand):
    data/sft_train.jsonl   1,500 Alpaca rows
    data/eval.jsonl           64 Alpaca rows, disjoint from training
    data/dpo_train.jsonl     256 UltraFeedback pairs
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset  # noqa: E402

SEED = 42
SFT_TRAIN_SIZE = 1500
EVAL_SIZE = 64
DPO_SIZE = 256
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {path}")


def main():
    rng = random.Random(SEED)

    # ---- Alpaca: SFT train + held-out eval -------------------------------
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    indices = list(range(len(alpaca)))
    rng.shuffle(indices)
    picked = [alpaca[i] for i in indices[: SFT_TRAIN_SIZE + EVAL_SIZE]]
    rows = [
        {
            "instruction": r["instruction"],
            "input": r["input"],
            "output": r["output"],
        }
        for r in picked
        if r["output"].strip()
    ]
    train_rows = rows[:SFT_TRAIN_SIZE]
    eval_rows = rows[SFT_TRAIN_SIZE : SFT_TRAIN_SIZE + EVAL_SIZE]
    write_jsonl(train_rows, DATA_DIR / "sft_train.jsonl")
    write_jsonl(eval_rows, DATA_DIR / "eval.jsonl")

    # ---- UltraFeedback binarized: DPO pairs ------------------------------
    # Streamed and truncated: the full split is ~61k pairs, we take 256.
    stream = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized", split="train_prefs", streaming=True
    )
    pairs = []
    for row in stream:
        chosen = next(
            (m["content"] for m in row["chosen"] if m["role"] == "assistant"), None
        )
        rejected = next(
            (m["content"] for m in row["rejected"] if m["role"] == "assistant"), None
        )
        if not chosen or not rejected or not row["prompt"].strip():
            continue
        pairs.append(
            {"prompt": row["prompt"], "chosen": chosen, "rejected": rejected}
        )
        if len(pairs) >= DPO_SIZE:
            break
    write_jsonl(pairs, DATA_DIR / "dpo_train.jsonl")

    meta = {
        "seed": SEED,
        "sft_source": "tatsu-lab/alpaca (train split, 52,002 rows), CC BY-NC 4.0",
        "sft_train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "eval_disjoint_from_train": True,
        "dpo_source": "HuggingFaceH4/ultrafeedback_binarized (train_prefs), "
        "first 256 usable pairs from a stream",
        "dpo_size": len(pairs),
    }
    (DATA_DIR / "data_card.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
