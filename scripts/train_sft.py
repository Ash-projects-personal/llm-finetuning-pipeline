"""Supervised fine-tuning with LoRA on the Alpaca subsample.

Trains HuggingFaceTB/SmolLM2-135M on data/sft_train.jsonl (see
scripts/prepare_data.py) on whatever hardware is present. On the
machine this was developed on (2 CPU cores, no GPU) one run takes
tens of minutes, and the run log in outputs/ says exactly what the
run cost and how the loss moved.

    python scripts/prepare_data.py   # first, once
    python scripts/train_sft.py

Writes:
    outputs/sft_adapter/       LoRA adapter + tokenizer
    outputs/sft_merged/        adapter merged into the base weights
    outputs/sft_training_log.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.common import RunLog, build_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
# Local copy of the model (scripts/download_model.sh). Used when present
# so runs do not depend on the Hugging Face cache.
MODEL_PATH = str(ROOT / "models" / "SmolLM2-135M")
MAX_LEN = 512
SEED = 42

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
TRAIN = dict(per_device_train_batch_size=4, gradient_accumulation_steps=2,
             num_train_epochs=2, learning_rate=2e-4, warmup_steps=11,
             logging_steps=10)


def load_rows():
    rows = []
    with open(ROOT / "data" / "sft_train.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def tokenize_rows(rows, tokenizer):
    """Tokenize prompt and full text; mask everything before the response."""
    features = []
    skipped = 0
    eos_id = tokenizer.eos_token_id
    for row in rows:
        prompt = build_prompt(row["instruction"], row.get("input", ""))
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt + row["output"], add_special_tokens=False)[
            "input_ids"
        ] + [eos_id]
        if len(full_ids) > MAX_LEN:
            skipped += 1
            continue
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        features.append({"input_ids": full_ids, "labels": labels})
    return features, skipped


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attention = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = maxlen - n
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attention.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention),
        }


def main():
    log = RunLog("sft_lora")
    rows = load_rows()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    features, skipped = tokenize_rows(rows, tokenizer)
    print(f"training examples: {len(features)} (skipped {skipped} over {MAX_LEN} tokens)")

    # fp32 on purpose: on CPU it is faster than bf16 at this model size.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32
    )
    model.config.use_cache = False
    peft_cfg = LoraConfig(task_type="CAUSAL_LM", **LORA)
    model = get_peft_model(model, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} of {total:,}")

    args = TrainingArguments(
        output_dir=str(ROOT / "outputs" / "sft_adapter"),
        seed=SEED,
        report_to=[],
        # Checkpoint every 50 steps so an interrupted run can resume
        # instead of starting over (see the resume logic below).
        save_strategy="steps",
        save_steps=50,
        dataloader_num_workers=0,
        **TRAIN,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(features),
        data_collator=Collator(tokenizer.pad_token_id),
    )
    checkpoints = sorted(
        (ROOT / "outputs" / "sft_adapter").glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    resumed_from = None
    if checkpoints:
        resumed_from = str(checkpoints[-1])
        print(f"resuming from {resumed_from}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(str(ROOT / "outputs" / "sft_adapter"))
    tokenizer.save_pretrained(str(ROOT / "outputs" / "sft_adapter"))

    merged = model.merge_and_unload()
    merged.save_pretrained(str(ROOT / "outputs" / "sft_merged"))
    tokenizer.save_pretrained(str(ROOT / "outputs" / "sft_merged"))

    loss_points = [
        {"step": h["step"], "loss": h.get("loss"), "epoch": h.get("epoch")}
        for h in trainer.state.log_history
        if "loss" in h
    ]
    log.add(
        model=MODEL_ID,
        model_path=MODEL_PATH,
        dataset="tatsu-lab/alpaca subsample (data/sft_train.jsonl)",
        train_examples=len(features),
        skipped_too_long=skipped,
        max_len=MAX_LEN,
        lora=LORA,
        train_args=TRAIN,
        trainable_params=trainable,
        total_params=total,
        resumed_from_checkpoint=resumed_from,
        loss_points=loss_points,
        final_loss=loss_points[-1]["loss"] if loss_points else None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_threads=torch.get_num_threads(),
    )
    out = log.finish(ROOT / "outputs" / "sft_training_log.json")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
