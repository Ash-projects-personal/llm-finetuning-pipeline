"""Evaluate a model on the held-out Alpaca eval set, for real.

Loads a model, generates an answer for every prompt in
data/eval.jsonl, and scores the generations against the reference
answers with ROUGE-L and normalized exact match (src/common.py).
Predictions are saved verbatim, so every number can be rechecked.

    python scripts/evaluate.py --tag base   # the untuned base model
    python scripts/evaluate.py --tag sft    # outputs/sft_merged
    python scripts/evaluate.py --tag dpo    # outputs/dpo_merged

Writes outputs/predictions/<tag>.jsonl, outputs/metrics/<tag>.json,
and refreshes outputs/metrics_summary.json with every tag present.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.common import (  # noqa: E402
    RunLog,
    build_prompt,
    clean_generation,
    exact_match,
    rouge_l_f1,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
PATHS = {
    "base": str(ROOT / "models" / "SmolLM2-135M"),
    "sft": str(ROOT / "outputs" / "sft_merged"),
    "dpo": str(ROOT / "outputs" / "dpo_merged"),
}
MAX_NEW_TOKENS = 48


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=list(PATHS))
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    model_path = args.model_path or PATHS[args.tag]

    log = RunLog(f"eval_{args.tag}")
    rows = [json.loads(l) for l in open(ROOT / "data" / "eval.jsonl")]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # fp32 on purpose: on CPU it is faster than bf16 for this model size
    # and it keeps every evaluation on identical numerics.
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32
    )
    model.eval()

    pred_dir = ROOT / "outputs" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{args.tag}.jsonl"

    rouge_sum, em_sum = 0.0, 0.0
    with open(pred_path, "w") as out:
        for i, row in enumerate(rows):
            prompt = build_prompt(row["instruction"], row.get("input", ""))
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            with torch.no_grad():
                gen = model.generate(
                    **ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = gen[0][ids["input_ids"].shape[1]:]
            prediction = clean_generation(
                tokenizer.decode(new_tokens, skip_special_tokens=True)
            )
            r = rouge_l_f1(prediction, row["output"])
            e = exact_match(prediction, row["output"])
            rouge_sum += r
            em_sum += e
            out.write(json.dumps({
                "instruction": row["instruction"],
                "input": row.get("input", ""),
                "reference": row["output"],
                "prediction": prediction,
                "rouge_l_f1": round(r, 4),
                "exact_match": e,
            }, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{args.tag}] {i + 1}/{len(rows)} rouge_l={r:.3f}")

    n = len(rows)
    metrics = {
        "tag": args.tag,
        "model_path": model_path,
        "eval_set": "data/eval.jsonl (held-out Alpaca subsample)",
        "n": n,
        "rouge_l_f1_mean": round(rouge_sum / n, 4),
        "exact_match_mean": round(em_sum / n, 4),
        "decoding": "greedy",
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    log.add(**metrics)
    log.finish(ROOT / "outputs" / "metrics" / f"{args.tag}.json")
    print(json.dumps(metrics, indent=2))

    # Refresh the combined summary with every tag evaluated so far.
    summary = {}
    for f in sorted((ROOT / "outputs" / "metrics").glob("*.json")):
        d = json.loads(f.read_text())
        summary[d["tag"]] = {
            "rouge_l_f1_mean": d["rouge_l_f1_mean"],
            "exact_match_mean": d["exact_match_mean"],
            "n": d["n"],
            "model_path": d["model_path"],
        }
    (ROOT / "outputs" / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
