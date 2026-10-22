import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import build_prompt, clean_generation, exact_match, rouge_l_f1


def test_prompt_without_input():
    p = build_prompt("Say hello")
    assert p == "### Instruction:\nSay hello\n\n### Response:\n"


def test_prompt_with_input():
    p = build_prompt("Translate", "bonjour")
    assert "### Input:\nbonjour" in p
    assert p.endswith("### Response:\n")


def test_rouge_identical():
    assert rouge_l_f1("the cat sat", "the cat sat") == 1.0


def test_rouge_disjoint():
    assert rouge_l_f1("alpha beta", "gamma delta") == 0.0


def test_rouge_partial():
    # LCS is 2 of 3 tokens on both sides -> F1 = 2/3
    assert abs(rouge_l_f1("the cat sat", "the cat ran") - 2 / 3) < 1e-9


def test_rouge_empty():
    assert rouge_l_f1("", "anything") == 0.0
    assert rouge_l_f1("anything", "") == 0.0


def test_exact_match_normalization():
    assert exact_match("The Cat!", "the cat") == 1.0
    assert exact_match("the cat", "the dog") == 0.0


def test_clean_generation_cuts_next_section():
    text = "Some answer.\n### Instruction:\nAnother question"
    assert clean_generation(text) == "Some answer."
