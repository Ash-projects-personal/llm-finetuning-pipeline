"""
Dataset format validator tests.

The training/DPO entry points load JSONL with strict shape requirements:

  - SFT (``train.py``):    each line is ``{"instruction", "input"?, "output"}``
  - DPO (``dpo_train.py``): each line is ``{"prompt", "chosen", "rejected"}``

These tests pin those contracts at the helper level so a malformed dataset
fails loudly *before* an expensive GPU launch.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ── helper: shipped sample dataset ───────────────────────────────────────


REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SFT = REPO_ROOT / "data" / "sample_train.jsonl"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            items.append(json.loads(stripped))
    return items


# ── SFT validator (mirrors train.load_dataset + format_instruction) ──────


def _validate_sft_record(rec: Dict[str, Any]) -> None:
    if not isinstance(rec, dict):
        raise TypeError(f"Record must be dict, got {type(rec).__name__}")
    if "instruction" not in rec or "output" not in rec:
        raise ValueError("Missing required keys: 'instruction' and/or 'output'")
    if not isinstance(rec["instruction"], str) or not rec["instruction"].strip():
        raise ValueError("'instruction' must be a non-empty string")
    if not isinstance(rec["output"], str) or not rec["output"].strip():
        raise ValueError("'output' must be a non-empty string")
    if "input" in rec and not isinstance(rec["input"], str):
        raise ValueError("'input' must be a string when present")


# ── DPO validator (mirrors dpo_train.load_preference_data) ───────────────


def _validate_dpo_record(rec: Dict[str, Any]) -> None:
    for key in ("prompt", "chosen", "rejected"):
        if key not in rec:
            raise ValueError(f"Missing required key: {key!r}")
        if not isinstance(rec[key], str) or not rec[key].strip():
            raise ValueError(f"{key!r} must be a non-empty string")
    if rec["chosen"] == rec["rejected"]:
        raise ValueError("'chosen' and 'rejected' must differ")


# ── shipped sample data ──────────────────────────────────────────────────


class TestShippedSampleDataset:
    def test_sample_train_jsonl_exists(self) -> None:
        assert SAMPLE_SFT.exists(), "data/sample_train.jsonl should be tracked"

    def test_every_line_is_valid_json_object(self) -> None:
        records = _read_jsonl(SAMPLE_SFT)
        assert len(records) >= 1
        for r in records:
            assert isinstance(r, dict)

    def test_every_record_passes_sft_validator(self) -> None:
        for r in _read_jsonl(SAMPLE_SFT):
            _validate_sft_record(r)


# ── SFT validator ────────────────────────────────────────────────────────


class TestSFTValidator:
    def test_minimal_valid_record(self) -> None:
        _validate_sft_record({"instruction": "do x", "output": "x done"})

    def test_with_input_field(self) -> None:
        _validate_sft_record(
            {"instruction": "translate", "input": "hello", "output": "bonjour"}
        )

    @pytest.mark.parametrize(
        "bad",
        [
            {},
            {"instruction": "no output"},
            {"output": "no instruction"},
            {"instruction": "", "output": "blank instruction"},
            {"instruction": "blank output", "output": ""},
        ],
    )
    def test_rejects_missing_or_empty_required_fields(self, bad: Dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            _validate_sft_record(bad)

    def test_rejects_non_string_instruction(self) -> None:
        with pytest.raises(ValueError):
            _validate_sft_record({"instruction": 42, "output": "x"})

    def test_rejects_non_string_input_field(self) -> None:
        with pytest.raises(ValueError):
            _validate_sft_record({"instruction": "a", "input": 7, "output": "b"})

    def test_rejects_non_dict_record(self) -> None:
        with pytest.raises(TypeError):
            _validate_sft_record(["not", "a", "dict"])  # type: ignore[arg-type]


# ── format_instruction (imported from train.py) ──────────────────────────


class TestFormatInstruction:
    def test_alpaca_format_no_input(self) -> None:
        from train import format_instruction

        out = format_instruction({"instruction": "Greet the user", "output": "Hi!"})
        assert "### Instruction:\nGreet the user" in out
        assert "### Response:\nHi!" in out
        assert "### Input:" not in out

    def test_alpaca_format_with_input(self) -> None:
        from train import format_instruction

        out = format_instruction(
            {"instruction": "Translate", "input": "hello", "output": "bonjour"}
        )
        assert "### Instruction:\nTranslate" in out
        assert "### Input:\nhello" in out
        assert "### Response:\nbonjour" in out

    def test_missing_fields_default_to_empty_string(self) -> None:
        from train import format_instruction

        # The helper uses .get(...) with "" default; should not crash.
        out = format_instruction({"instruction": "only instr"})
        assert "### Instruction:\nonly instr" in out
        assert "### Response:\n" in out


# ── load_dataset round-trip ──────────────────────────────────────────────


class TestLoadDataset:
    def test_loads_jsonl_file(self, tmp_path: Path) -> None:
        from train import load_dataset

        path = tmp_path / "tiny.jsonl"
        path.write_text(
            "\n".join(
                json.dumps({"instruction": f"q{i}", "output": f"a{i}"})
                for i in range(4)
            )
            + "\n"
        )
        items = load_dataset(str(path))
        assert len(items) == 4
        assert items[0] == {"instruction": "q0", "output": "a0"}

    def test_load_dataset_blank_lines_raise(self, tmp_path: Path) -> None:
        """A blank line inside the JSONL is corrupt — should surface a clear
        JSON error, not silently produce empty dicts."""
        from train import load_dataset

        path = tmp_path / "broken.jsonl"
        path.write_text(
            textwrap.dedent(
                """\
                {"instruction": "ok", "output": "ok"}

                {"instruction": "ok2", "output": "ok2"}
                """
            )
        )
        with pytest.raises(json.JSONDecodeError):
            load_dataset(str(path))


# ── DPO validator ────────────────────────────────────────────────────────


class TestDPOValidator:
    def test_valid_preference_pair(self) -> None:
        _validate_dpo_record(
            {"prompt": "Hi", "chosen": "Hello!", "rejected": "go away"}
        )

    @pytest.mark.parametrize("missing", ["prompt", "chosen", "rejected"])
    def test_rejects_missing_field(self, missing: str) -> None:
        rec = {"prompt": "p", "chosen": "c", "rejected": "r"}
        rec.pop(missing)
        with pytest.raises(ValueError, match=missing):
            _validate_dpo_record(rec)

    def test_rejects_identical_chosen_and_rejected(self) -> None:
        with pytest.raises(ValueError, match="must differ"):
            _validate_dpo_record(
                {"prompt": "p", "chosen": "same", "rejected": "same"}
            )

    def test_rejects_empty_string_fields(self) -> None:
        with pytest.raises(ValueError):
            _validate_dpo_record({"prompt": "p", "chosen": "", "rejected": "r"})


# ── ROUGE-L metric (lightweight, no transformers needed) ─────────────────


class TestRougeL:
    def test_identical_strings_score_one(self) -> None:
        from evaluate import compute_rouge_l

        assert compute_rouge_l("a quick brown fox", "a quick brown fox") == 1.0

    def test_disjoint_strings_score_zero(self) -> None:
        from evaluate import compute_rouge_l

        assert compute_rouge_l("the cat sat", "lorem ipsum dolor") == 0.0

    def test_partial_overlap_in_unit_range(self) -> None:
        from evaluate import compute_rouge_l

        score = compute_rouge_l(
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox",
        )
        assert 0.0 < score < 1.0

    def test_empty_inputs_short_circuit(self) -> None:
        from evaluate import compute_rouge_l

        assert compute_rouge_l("", "anything") == 0.0
        assert compute_rouge_l("anything", "") == 0.0
