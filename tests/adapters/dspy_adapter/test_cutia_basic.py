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
        track_stats=True,
        num_threads=1,  # Single-threaded for deterministic test
        num_candidates=1,  # Single candidate for speed
        traversal_strategy="post_order",
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
        )
        assert optimizer.traversal_strategy == strategy


def test_cutia_stats_thread_safety():
    """Test that stats increments are thread-safe under concurrent access."""
    import threading

    lm = DummyLM(["test"])

    optimizer = CUTIA(
        prompt_model=lm,
        task_model=lm,
        metric=simple_metric,
    )

    # Reset stats to known state
    optimizer.stats = {"nodes_visited": 0, "nodes_cut": 0, "nodes_rewritten": 0, "llm_calls": 0}

    num_threads = 8
    increments_per_thread = 100

    def increment_all_stats():
        for _ in range(increments_per_thread):
            optimizer._increment_stat("llm_calls")
            optimizer._increment_stat("nodes_visited")
            optimizer._increment_stat("nodes_cut")
            optimizer._increment_stat("nodes_rewritten")

    threads = [threading.Thread(target=increment_all_stats) for _ in range(num_threads)]

    # Start all threads
    for t in threads:
        t.start()

    # Wait for completion
    for t in threads:
        t.join()

    expected = num_threads * increments_per_thread

    # All counters should have the exact expected count (no lost increments)
    assert optimizer.stats["llm_calls"] == expected, (
        f"llm_calls: expected {expected}, got {optimizer.stats['llm_calls']} "
    )
    assert optimizer.stats["nodes_visited"] == expected, (
        f"nodes_visited: expected {expected}, got {optimizer.stats['nodes_visited']}"
    )
    assert optimizer.stats["nodes_cut"] == expected, (
        f"nodes_cut: expected {expected}, got {optimizer.stats['nodes_cut']}"
    )
    assert optimizer.stats["nodes_rewritten"] == expected, (
        f"nodes_rewritten: expected {expected}, got {optimizer.stats['nodes_rewritten']}"
    )
