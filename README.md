# LLM fine tuning pipeline

I built this project to learn what fine tuning actually does to a small language model, with every step run and measured by me on my own machine. It takes one small open model through supervised fine tuning and then a short DPO run, and it evaluates all three versions (base, SFT, DPO) on the same held out set with the same metric.

The scale is small on purpose. I had no GPU for this, only a two core CPU machine with about 7 GB of RAM. So I picked a 135M parameter model, subsampled public datasets, and kept the runs short enough to finish. Everything below comes out of the run logs in `outputs/`. If a number is not in a log file, I do not claim it.

## Setup

**Model.** HuggingFaceTB/SmolLM2-135M, 136,358,208 parameters in total. I train in fp32 because on this CPU it is faster than bf16 at this model size.

**SFT data.** tatsu-lab/alpaca, train split with 52,002 rows, license CC BY-NC 4.0. I used a random subsample of 1,500 rows for training and 62 separate rows as the held out eval set (seed 42). The eval instructions do not overlap the training instructions. `scripts/prepare_data.py` writes both files and `data/data_card.json` records the counts.

**DPO data.** HuggingFaceH4/ultrafeedback_binarized, train_prefs split. I used the first 256 usable preference pairs from a stream. Each pair has one prompt, one chosen response, and one rejected response.

**Hardware.** 2 CPU cores, no GPU, about 7 GB RAM. Python 3.12, torch 2.14.1 (CPU wheel), transformers 5.20.2, peft 0.21.2, trl 1.16.0. Exact versions are in `requirements.txt`.

## What I ran

**SFT.** LoRA adapters (rank 16, alpha 32, dropout 0.05) on the q, k, v, and o projections. That is 1,843,200 trainable parameters, about 1.4 percent of the model. Two epochs over the 1,500 examples, batch size 4 with gradient accumulation 2, learning rate 2e-4, max sequence length 512, loss only on the response tokens. 376 steps in total. Loss went from 1.879 at step 10 to 1.653 at step 370, with an average of 1.716 across the run. The Trainer recorded 6,602 seconds of training time, about 1 hour 50 minutes.

**DPO.** Started from the merged SFT model, fresh LoRA adapter with the same shape, TRL DPOTrainer, beta 0.1, learning rate 1e-4, one epoch over the 256 pairs, 32 steps. The training log is in `outputs/dpo_training_log.json`. Honest summary: it did not learn much. The loss stayed near 0.70 and the reward accuracy stayed near chance, ending at 0.40 on the last logged point. Peak memory measured in the final segment was 3,585 MB.

**Evaluation.** Greedy decoding, at most 48 new tokens, same prompt format for all three models, on the 62 held out Alpaca examples. Metrics are ROUGE-L F1 (longest common subsequence based, implemented in `src/common.py`) and exact match after simple normalization. Predictions for every example are saved in `outputs/predictions/`.

## Results

| Model | ROUGE-L F1 (mean) | Exact match (mean) |
|---|---|---|
| Base SmolLM2-135M | 0.131 | 0.000 |
| After SFT | 0.228 | 0.016 |
| After SFT + DPO | 0.227 | 0.016 |

Full per model files are in `outputs/metrics/`, and `outputs/metrics_summary.json` collects the three rows.

What I take from this:

- SFT helped in a clear and measurable way. ROUGE-L rose from 0.131 to 0.228, and exact match went from 0 of 62 to 1 of 62. Reading the saved predictions matches the scores. The base model rambles and often misses the instruction. The SFT model answers the instruction directly and stops.
- DPO at this scale changed nothing I can measure. The eval scores after DPO (0.227 and 0.016) are the same as after SFT within noise, and its own training metrics explain why: the reward accuracy never got above chance for long. 256 pairs and 32 steps on a 135M model is most likely too little signal. I am reporting it as run, not dressing it up.

## Limits

- 62 eval examples is a small set. These scores are a rough signal, not a benchmark result.
- ROUGE-L measures word overlap with one reference answer. It rewards matching the reference style, which is part of what SFT teaches here. It does not measure correctness or usefulness in general.
- One small model, one dataset, one seed. Nothing here generalizes beyond this setup.

## One honest incident note

The machine restarted near the end of the SFT run, after training had completed all 376 steps but during the final merged model save. The merged model and `outputs/sft_training_log.json` were regenerated from the final checkpoint and the saved adapter, with no retraining. The log file itself carries a note explaining this, which is also why peak memory for SFT training is not reported: the process that could measure it was killed before it wrote the log. DPO training also resumed once from a checkpoint after a restart, and its log records that too.

## Files

- `scripts/prepare_data.py` builds the datasets in `data/` and writes `data/data_card.json`.
- `scripts/train_sft.py` trains the LoRA adapter and writes `outputs/sft_training_log.json`, the adapter, and the merged model.
- `scripts/train_dpo.py` runs DPO from the SFT model and writes `outputs/dpo_training_log.json`.
- `scripts/evaluate.py` evaluates one model version (`--tag base`, `--tag sft`, or `--tag dpo`) and writes predictions plus metrics into `outputs/`.
- `scripts/download_model.sh` fetches the base model into `models/`.
- `src/common.py` holds the shared prompt format and metric code.
- `tests/test_common.py` tests the prompt builder and metrics. Run with `python -m pytest tests/`.

## Reproduce

```bash
pip install -r requirements.txt
bash scripts/download_model.sh
python scripts/prepare_data.py
python scripts/train_sft.py
python scripts/evaluate.py --tag base
python scripts/evaluate.py --tag sft
python scripts/train_dpo.py
python scripts/evaluate.py --tag dpo
```

On a two core CPU, expect roughly 2 hours for SFT, about 1.5 hours for DPO, and a few minutes to about 25 minutes per evaluation. On a GPU all of this is much faster, but then the timing numbers in my logs would not match yours, which is fine. The point is the pipeline and the method, not the exact scores.
