"""
pytest configuration for llm-finetuning-pipeline tests.

Heavy deps (transformers, peft, trl, torch, bitsandbytes) are NOT installed
in CI — they're swapped via ``pytest.importorskip`` per-test. The test
modules only exercise pure-Python helpers (dataset format, prompt
formatting, LoRA-config validation, ROUGE-L) that don't need them.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
