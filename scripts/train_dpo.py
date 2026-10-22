"""Direct Preference Optimization on the UltraFeedback subsample.

Starts from the SFT model (outputs/sft_merged, produced by
train_sft.py) and trains a fresh LoRA adapter on data/dpo_train.jsonl
with TRL's DPOTrainer. The reference model is the SFT model itself
(TRL handles this by disabling the adapter for reference passes).

    python scripts/train_dpo.py

Writes:
    outputs/dpo_adapter/       LoRA adapter + tokenizer
    outputs/dpo_merged/        adapter merged into the SFT weights
    outputs/dpo_training_log.json   (loss and reward accuracy per step)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import DPOConfig, DPOTrainer  # noqa: E402

from src.common import RunLog  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SFT_MODEL = str(ROOT / "outputs" / "sft_merged")
SEED = 42

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
DPO = dict(beta=0.1, learning_rate=1e-4, per_device_train_batch_size=2,
           gradient_accumulation_steps=4, num_train_epochs=1,
           max_length=512, logging_steps=2)


def main():
    log = RunLog("dpo_lora")
    rows = [json.loads(l) for l in open(ROOT / "data" / "dpo_train.jsonl")]
    dataset = Dataset.from_list(rows)
    print(f"preference pairs: {len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # fp32 on purpose: on CPU it is faster than bf16 at this model size.
    model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL, dtype=torch.float32
    )
    model.config.use_cache = False

    cfg = DPOConfig(
        output_dir=str(ROOT / "outputs" / "dpo_adapter"),
        seed=SEED,
        use_cpu=True,
        report_to=[],
        # Checkpoint every 8 steps so an interrupted run can resume
        # instead of starting over (see the resume logic below).
        save_strategy="steps",
        save_steps=8,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        **DPO,
    )
    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(task_type="CAUSAL_LM", **LORA),
    )
    checkpoints = sorted(
        (ROOT / "outputs" / "dpo_adapter").glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    resumed_from = None
    if checkpoints:
        resumed_from = str(checkpoints[-1])
        print(f"resuming from {resumed_from}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(str(ROOT / "outputs" / "dpo_adapter"))
    tokenizer.save_pretrained(str(ROOT / "outputs" / "dpo_adapter"))
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(ROOT / "outputs" / "dpo_merged"))
    tokenizer.save_pretrained(str(ROOT / "outputs" / "dpo_merged"))

    points = []
    for h in trainer.state.log_history:
        if "loss" in h:
            points.append({
                "step": h["step"],
                "loss": h.get("loss"),
                "rewards_accuracies": h.get("rewards/accuracies"),
                "rewards_margins": h.get("rewards/margins"),
            })
    log.add(
        base_model=SFT_MODEL,
        dataset="HuggingFaceH4/ultrafeedback_binarized subsample "
                "(data/dpo_train.jsonl)",
        pairs=len(dataset),
        lora=LORA,
        dpo_args={k: DPO[k] for k in ("beta", "learning_rate",
                                      "num_train_epochs", "max_length")},
        resumed_from_checkpoint=resumed_from,
        log_points=points,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_threads=torch.get_num_threads(),
    )
    out = log.finish(ROOT / "outputs" / "dpo_training_log.json")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
