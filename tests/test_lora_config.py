"""
LoRA-config sanity tests.

QLoRA fine-tuning is very sensitive to:
  - r (rank) and alpha scaling — the effective scaling factor is alpha / r,
    typically 2.0 (so alpha = 2*r is the conventional default).
  - dropout in a safe range (0 .. 0.5).
  - target_modules containing the attention projections for transformer
    fine-tuning to actually work.

These tests pin the defaults declared in ``train.TrainingConfig`` and
``dpo_train.DPOConfig``, and they also exercise constructor overrides so
the configs behave as plain dataclasses (no validation surprises later).

They do NOT import peft / transformers / trl — the configs are pure
dataclasses with no heavy dependency surface.
"""
from __future__ import annotations

import pytest

from train import TrainingConfig
from dpo_train import DPOConfig


# ── default LoRA / QLoRA values ──────────────────────────────────────────


class TestTrainingConfigLoRADefaults:
    def test_rank_is_power_of_two_in_safe_range(self) -> None:
        cfg = TrainingConfig()
        assert cfg.lora_r >= 4
        assert cfg.lora_r <= 128
        assert (cfg.lora_r & (cfg.lora_r - 1)) == 0, "rank should be a power of two"

    def test_alpha_is_twice_the_rank(self) -> None:
        """Convention: alpha = 2 * r gives an effective scaling factor of 2.0
        in the LoRA delta-W = (alpha / r) * B @ A. Drifting from that needs
        an intentional override."""
        cfg = TrainingConfig()
        assert cfg.lora_alpha == 2 * cfg.lora_r
        assert cfg.lora_alpha / cfg.lora_r == pytest.approx(2.0)

    def test_dropout_is_inside_unit_interval(self) -> None:
        cfg = TrainingConfig()
        assert 0.0 <= cfg.lora_dropout <= 0.5

    def test_target_modules_cover_attention_projections(self) -> None:
        cfg = TrainingConfig()
        targets = set(cfg.target_modules)
        # at minimum we need Q and V — K and O are nice-to-have.
        required = {"q_proj", "v_proj"}
        assert required.issubset(targets), f"missing {required - targets}"
        # and absolutely nothing weird is included by default
        allowed = {"q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"}
        assert targets.issubset(allowed), f"unexpected modules: {targets - allowed}"

    def test_target_modules_default_list_is_not_shared_across_instances(self) -> None:
        """Default-factory pitfall: every TrainingConfig must get its own list."""
        a, b = TrainingConfig(), TrainingConfig()
        a.target_modules.append("gate_proj")
        assert "gate_proj" not in b.target_modules


# ── 4-bit / NF4 quantization defaults ────────────────────────────────────


class TestQuantizationDefaults:
    def test_load_in_4bit_enabled(self) -> None:
        assert TrainingConfig().load_in_4bit is True

    def test_quant_type_is_nf4(self) -> None:
        assert TrainingConfig().bnb_4bit_quant_type == "nf4"

    def test_double_quant_enabled(self) -> None:
        assert TrainingConfig().use_double_quant is True

    def test_compute_dtype_is_a_valid_torch_dtype_name(self) -> None:
        # We don't import torch (CI doesn't have it), but the string must be
        # plausible — match the production fp16/bf16/fp32 vocabulary.
        assert TrainingConfig().bnb_4bit_compute_dtype in {
            "float16", "bfloat16", "float32",
        }


# ── training hyperparameter sanity ───────────────────────────────────────


class TestTrainingHyperparamSanity:
    def test_epochs_and_batch_size_positive(self) -> None:
        cfg = TrainingConfig()
        assert cfg.num_train_epochs >= 1
        assert cfg.per_device_train_batch_size >= 1
        assert cfg.gradient_accumulation_steps >= 1

    def test_effective_batch_size_matches_expected(self) -> None:
        """effective = per_device_batch * grad_accum (per-GPU)."""
        cfg = TrainingConfig()
        effective = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
        assert effective == 16  # 4 * 4 — the documented default

    def test_warmup_ratio_inside_unit_interval(self) -> None:
        assert 0.0 < TrainingConfig().warmup_ratio < 1.0

    def test_learning_rate_in_qlora_range(self) -> None:
        # QLoRA fine-tuning typically uses 1e-4 .. 5e-4. Tight bounds so a
        # bad copy-paste from a full-fine-tune script gets caught.
        lr = TrainingConfig().learning_rate
        assert 1e-5 <= lr <= 5e-4

    def test_lr_scheduler_supported(self) -> None:
        assert TrainingConfig().lr_scheduler_type in {
            "linear", "cosine", "cosine_with_restarts", "constant",
            "constant_with_warmup", "polynomial",
        }

    def test_max_seq_length_is_power_of_two(self) -> None:
        msl = TrainingConfig().max_seq_length
        assert 512 <= msl <= 8192
        assert (msl & (msl - 1)) == 0


# ── DPOConfig sanity ─────────────────────────────────────────────────────


class TestDPOConfigSanity:
    def test_default_beta_in_recommended_range(self) -> None:
        """DPO paper recommends beta in [0.1, 0.5]."""
        assert 0.05 <= DPOConfig().beta <= 0.5

    def test_dpo_learning_rate_is_small(self) -> None:
        """DPO uses a much smaller LR than SFT — should be well below 1e-5."""
        assert DPOConfig().learning_rate <= 1e-5

    def test_prompt_length_fits_inside_max_length(self) -> None:
        cfg = DPOConfig()
        assert cfg.max_prompt_length < cfg.max_length

    def test_constructor_overrides_propagate(self) -> None:
        cfg = DPOConfig(beta=0.2, output_dir="/tmp/dpo-out")
        assert cfg.beta == pytest.approx(0.2)
        assert cfg.output_dir == "/tmp/dpo-out"
        # untouched defaults remain
        assert cfg.dataset_path == "data/dpo_preferences.jsonl"


# ── constructor override behaviour ───────────────────────────────────────


class TestTrainingConfigOverrides:
    def test_rank_alpha_override(self) -> None:
        cfg = TrainingConfig(lora_r=8, lora_alpha=16)
        assert cfg.lora_r == 8
        assert cfg.lora_alpha == 16
        assert cfg.lora_alpha / cfg.lora_r == pytest.approx(2.0)

    def test_target_modules_override(self) -> None:
        cfg = TrainingConfig(target_modules=["q_proj", "v_proj"])
        assert cfg.target_modules == ["q_proj", "v_proj"]

    def test_run_name_optional(self) -> None:
        assert TrainingConfig().run_name is None
        assert TrainingConfig(run_name="exp-7").run_name == "exp-7"
