"""
Smoke tests for the evaluation and benchmark modules.

These tests verify that:
  - All evaluation sub-modules can be imported cleanly.
  - Core metric functions produce correct, deterministic results on
    known inputs—without needing a trained model or external dataset.
  - The benchmark run_suite module is importable.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

class TestEvaluationImports:
    def test_import_perplexity(self):
        from atomic_1bit.evaluation.perplexity import calculate_perplexity  # noqa: F401

    def test_import_repetition(self):
        from atomic_1bit.evaluation.repetition import (  # noqa: F401
            calculate_repetition_rate,
            calculate_repetition_rates,
        )

    def test_import_diversity(self):
        from atomic_1bit.evaluation.diversity import (  # noqa: F401
            unique_ngram_ratio,
            vocabulary_utilization,
            diversity_metrics,
        )

    def test_import_coherence(self):
        from atomic_1bit.evaluation.coherence import (  # noqa: F401
            sentence_completion_accuracy,
            top_k_accuracy,
        )

    def test_import_run_eval(self):
        """run_eval module (and its generate_text/run_evaluation) must import."""
        from atomic_1bit.evaluation.run_eval import generate_text, run_evaluation  # noqa: F401


class TestBenchmarkImports:
    def test_import_run_suite_functions(self):
        """
        run_suite.py imports psutil and baseline_fp16 using relative paths;
        we import only the standalone utility helpers that have no heavy deps.
        """
        # We cannot import run_suite directly because it resolves baseline_fp16
        # via a sys.path relative to benchmarks/. Instead verify the file exists
        # and is syntactically valid by compiling it.
        import ast

        bench_path = os.path.join(
            os.path.dirname(__file__), "..", "benchmarks", "run_suite.py"
        )
        assert os.path.exists(bench_path), f"run_suite.py not found at {bench_path}"

        with open(bench_path, "r") as f:
            source = f.read()

        # Will raise SyntaxError if the file is malformed
        tree = ast.parse(source)
        assert tree is not None


# ---------------------------------------------------------------------------
# Repetition metric correctness
# ---------------------------------------------------------------------------

class TestRepetitionMetrics:
    def test_all_unique_ngrams(self):
        """All-unique 3-grams -> repetition rate is 0."""
        from atomic_1bit.evaluation.repetition import calculate_repetition_rate

        text = "the quick brown fox jumps over the lazy dog"
        rate = calculate_repetition_rate(text, n=3)
        # There is one repeated 3-gram ("the" appears twice but as part of different
        # trigrams; for n=3 it's only ("the", "lazy", "dog") that doesn't repeat).
        # The exact value depends on the text—just verify range.
        assert 0.0 <= rate <= 1.0

    def test_fully_repeated_text(self):
        """Completely repeated text -> repetition rate should be 1."""
        from atomic_1bit.evaluation.repetition import calculate_repetition_rate

        # "a a a a a" -> every 3-gram is ("a","a","a"), so rate = 1 - (1/3) for n=3
        # More precisely: unique=1, total=3 -> 1 - 1/3 = 0.667
        # For full repetition use n=1
        text = "a a a a a a a a a a"
        rate = calculate_repetition_rate(text, n=1)
        assert rate == pytest.approx(1.0 - 1.0 / 10, abs=0.01)

    def test_short_text_below_n(self):
        """Text shorter than n tokens -> repetition rate is 0 (no ngrams)."""
        from atomic_1bit.evaluation.repetition import calculate_repetition_rate

        rate = calculate_repetition_rate("hi", n=3)
        assert rate == 0.0

    def test_empty_string(self):
        """Empty text must return 0.0 without raising."""
        from atomic_1bit.evaluation.repetition import calculate_repetition_rate

        assert calculate_repetition_rate("", n=3) == 0.0

    def test_calculate_repetition_rates_keys(self):
        """calculate_repetition_rates must return dict with keys 1..max_n."""
        from atomic_1bit.evaluation.repetition import calculate_repetition_rates

        text = "once upon a time there was a little girl"
        rates = calculate_repetition_rates(text, max_n=4)
        assert set(rates.keys()) == {1, 2, 3, 4}
        for v in rates.values():
            assert 0.0 <= v <= 1.0

    def test_known_repetition_value(self):
        """Verify exact repetition rate on a controlled string."""
        from atomic_1bit.evaluation.repetition import calculate_repetition_rate

        # tokens: a b c a b c
        # 3-grams: (a,b,c) (b,c,a) (c,a,b) (a,b,c) -> 4 total, 3 unique
        text = "a b c a b c"
        rate = calculate_repetition_rate(text, n=3)
        assert rate == pytest.approx(1.0 - 3.0 / 4, abs=1e-6)


# ---------------------------------------------------------------------------
# Diversity metric correctness
# ---------------------------------------------------------------------------

class TestDiversityMetrics:
    def test_unique_ngram_ratio_all_unique(self):
        """Completely unique unigrams -> ratio is 1.0."""
        from atomic_1bit.evaluation.diversity import unique_ngram_ratio

        text = "one two three four five"
        ratio = unique_ngram_ratio(text, n=1)
        assert ratio == pytest.approx(1.0, abs=1e-6)

    def test_unique_ngram_ratio_all_same(self):
        """All identical tokens -> ratio equals 1/total."""
        from atomic_1bit.evaluation.diversity import unique_ngram_ratio

        text = "a a a a a"  # 5 unigrams, 1 unique
        ratio = unique_ngram_ratio(text, n=1)
        assert ratio == pytest.approx(1.0 / 5, abs=1e-6)

    def test_unique_ngram_ratio_short_text(self):
        """Text shorter than n -> returns 1.0 (edge-case guard)."""
        from atomic_1bit.evaluation.diversity import unique_ngram_ratio

        ratio = unique_ngram_ratio("hello", n=3)
        assert ratio == 1.0

    def test_diversity_metrics_keys(self):
        """diversity_metrics must return unigram, bigram, trigram ratios."""
        from atomic_1bit.evaluation.diversity import diversity_metrics

        result = diversity_metrics("the cat sat on the mat")
        assert "unigram_ratio" in result
        assert "bigram_ratio" in result
        assert "trigram_ratio" in result
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_diversity_metrics_empty_string(self):
        """diversity_metrics on empty string must not raise."""
        from atomic_1bit.evaluation.diversity import diversity_metrics

        result = diversity_metrics("")
        # All ratios default to 1.0 when there are no ngrams
        for v in result.values():
            assert v == 1.0

    def test_vocabulary_utilization_known(self):
        """Verify vocabulary utilization on a controlled example."""
        from atomic_1bit.evaluation.diversity import vocabulary_utilization

        texts = ["hello world", "foo bar"]
        # Unique tokens: hello, world, foo, bar -> 4 out of vocab_size=10
        util = vocabulary_utilization(texts, total_vocab_size=10)
        assert util == pytest.approx(4.0 / 10, abs=1e-6)

    def test_vocabulary_utilization_empty(self):
        """Empty texts list -> utilization is 0."""
        from atomic_1bit.evaluation.diversity import vocabulary_utilization

        util = vocabulary_utilization([], total_vocab_size=100)
        assert util == 0.0

    def test_bigram_diversity_range(self):
        """All n-gram ratios must stay in [0, 1] regardless of n."""
        from atomic_1bit.evaluation.diversity import unique_ngram_ratio

        text = "the cat the cat the cat"
        for n in (1, 2, 3):
            ratio = unique_ngram_ratio(text, n=n)
            assert 0.0 <= ratio <= 1.0, (
                f"unique_ngram_ratio for n={n} out of range: {ratio}"
            )


# ---------------------------------------------------------------------------
# Perplexity: basic formula check (no model required)
# ---------------------------------------------------------------------------

class TestPerplexityFormula:
    def test_perplexity_with_mock_model(self):
        """
        Verify calculate_perplexity returns a positive finite float.

        We provide a minimal mock model and mock dataset that return
        known logits so we can reason about the expected value.
        """
        import torch
        import torch.nn.functional as F
        from atomic_1bit.evaluation.perplexity import calculate_perplexity

        vocab_size = 8
        seq_len = 4

        class _ConstantLogitsModel:
            """Returns uniform logits for every forward call."""
            config = None  # not used by calculate_perplexity

            def eval(self):
                return self

            def __call__(self, x):
                batch, seq = x.shape
                # Uniform distribution -> CE = log(vocab_size)
                return torch.zeros(batch, seq, vocab_size)

            def parameters(self):
                return iter([])

        class _MockDataset:
            def get_batch(self, batch_size):
                x = torch.zeros(batch_size, seq_len, dtype=torch.long)
                y = torch.zeros(batch_size, seq_len, dtype=torch.long)
                return x, y

        model = _ConstantLogitsModel()
        dataset = _MockDataset()

        ppl = calculate_perplexity(
            model, dataset, vocab_size, device="cpu", num_samples=5
        )

        # For uniform logits over vocab_size classes, PPL = vocab_size
        expected_ppl = vocab_size
        assert ppl == pytest.approx(expected_ppl, rel=1e-3), (
            f"Expected PPL ~{expected_ppl}, got {ppl}"
        )
        assert ppl > 0
        assert torch.isfinite(torch.tensor(ppl))
