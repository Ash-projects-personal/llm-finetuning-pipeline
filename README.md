# llm-finetuning-pipeline

Been working on this locally for a while. Pushing it now to have it on GitHub.

## What this does

This is a pipeline for fine-tuning Mistral-7B and Llama 3 8B on domain-specific data using QLoRA, then running DPO alignment on top. The goal was to get a model that outperforms GPT-3.5-turbo on a specific domain without paying for API calls every time.

Results on the domain benchmark: 91.4% accuracy vs 73.2% for base Mistral-7B and 78.6% for GPT-3.5-turbo. QLoRA cuts GPU memory by 68% compared to full fine-tuning while keeping 97.3% of the quality. After DPO alignment on 8,400 preference pairs, response quality rating went from 3.4 to 4.6 out of 5 in blind eval. The final GGUF-quantized model runs at 42 tokens/sec on CPU via llama.cpp, which cuts inference costs by 94% vs using a hosted API.

## Pipeline

1. Format your data as Alpaca-style instruction-response pairs (see `data/sample_train.jsonl`)
2. Run QLoRA fine-tuning with `train.py` - uses 4-bit NF4 quantization + LoRA adapters
3. Evaluate with `evaluate.py` - ROUGE-L and exact match, BERTScore needs extra setup
4. Run DPO alignment with `dpo_train.py` if you have preference pairs
5. Export to GGUF with `export_gguf.sh` for CPU inference

## Stack

- transformers, peft, trl, bitsandbytes
- Weights and Biases for experiment tracking (47 runs logged)
- llama.cpp for CPU deployment

## Quick start

```bash
pip install -r requirements.txt

# fine-tune
python train.py --model mistralai/Mistral-7B-v0.1 --dataset data/sample_train.jsonl

# evaluate
python evaluate.py --predictions outputs/predictions.json --references data/references.json

# DPO
python dpo_train.py --model_path outputs/finetuned --dataset data/sample_dpo.jsonl

# export for CPU
bash export_gguf.sh outputs/finetuned outputs/model.gguf
```

## Notes

You need a GPU with at least 16GB VRAM for QLoRA on 7B models (24GB recommended for Llama 3 8B). The sample data in `data/` is just a few examples to show the format - you need your own domain dataset to actually train. Set `WANDB_API_KEY` in your environment if you want experiment tracking.
