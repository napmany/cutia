"""Basic smoke tests for CUTIA DSPy adapter."""

import dspy
from dspy import Example
from dspy.predict import Predict
from dspy.utils.dummies import DummyLM

from cutia.adapters.dspy_adapter import CUTIA


class SimpleModule(dspy.Module):
    """Simple test module with a single predictor."""

    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


def simple_metric(example, prediction, trace=None):
    """Basic exact match metric."""
    return example.output == prediction.output


def test_cutia_basic_compile():
    """Test that CUTIA can compile a simple program without errors.

    This is a smoke test to verify:
    1. CUTIA can be instantiated
    2. The compile workflow runs without crashes
    3. Basic tree building and node processing works
    """
    # Create simple student module
    student = SimpleModule("input -> output")

    # Configure DummyLM to avoid real API calls
    # DummyLM cycles through responses, so we need enough for all operations
    # We provide enough responses for tree building and multiple evaluations
    responses = []

    # Add responses for tree building (ProposeChunk) - happens for each candidate
    for _ in range(5):
        responses.extend(
            [
                "false",  # has_chunk - no split needed
                "No chunk to extract",  # chunk_reason
            ]
        )

    # Add many evaluation responses (program will be evaluated multiple times)
    for _ in range(100):
        responses.extend(["blue", "green"])

    lm = DummyLM(responses)

    # Configure DSPy settings
    dspy.settings.configure(lm=lm)

    # Create CUTIA optimizer with minimal configuration
    optimizer = CUTIA(
        prompt_model=lm,
        task_model=lm,
        metric=simple_metric,
        max_depth=1,  # Keep tree shallow for test
        min_chunk_chars=200,  # High threshold to avoid splitting
        quality_mode="strict",
        target_compression_ratio=0.5,
        verbose=False,  # Quiet during tests
        track_stats=True,
        num_threads=1,  # Single-threaded for deterministic test
        num_candidates=1,  # Single candidate for speed
        traversal_strategy="post_order",
        parallel_tree_building=False,  # Disable parallelism for deterministic test
        enable_cutting=True,
        rewrite_strategy="basic",
    )

    # Create minimal training and validation datasets
    trainset = [
        Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
        Example(input="What color is grass?", output="green").with_inputs("input"),
    ]

    valset = [
        Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
    ]

    # Run compilation
    compiled_program = optimizer.compile(
        student=student,
        trainset=trainset,
        valset=valset,
    )

    # Basic assertions
    assert compiled_program is not None
    assert isinstance(compiled_program, dspy.Module)

    # Verify compression stats were tracked
    assert hasattr(compiled_program, "compression_stats")
    assert "mode" in compiled_program.compression_stats
    assert compiled_program.compression_stats["mode"] == "CUTIA"


def test_cutia_quality_modes():
    """Test that different quality modes are accepted."""
    lm = DummyLM(["test"])

    # Test each quality mode
    for mode in ["strict", "balanced", "aggressive"]:
        optimizer = CUTIA(
            prompt_model=lm,
            task_model=lm,
            metric=simple_metric,
            quality_mode=mode,
            verbose=False,
        )
        assert optimizer.quality_mode == mode


def test_cutia_traversal_strategies():
    """Test that different traversal strategies are accepted."""
    lm = DummyLM(["test"])

    # Test each strategy
    for strategy in ["post_order", "pre_order", "random"]:
        optimizer = CUTIA(
            prompt_model=lm,
            task_model=lm,
            metric=simple_metric,
            traversal_strategy=strategy,
            verbose=False,
        )
        assert optimizer.traversal_strategy == strategy
