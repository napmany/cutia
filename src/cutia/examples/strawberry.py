import argparse
import os
from pathlib import Path
from typing import List, Tuple

import datasets
import dspy
import tiktoken
from dotenv import load_dotenv

from cutia.adapters.dspy_adapter import CUTIA

# Load .env from the same directory as this script
load_dotenv(Path(__file__).parent / ".env")


class CountLetters(dspy.Signature):
    """Count how many times a specific letter appears in a word."""

    word: str = dspy.InputField(desc="The word to analyze")
    letter: str = dspy.InputField(desc="The letter to count")
    count: int = dspy.OutputField(desc="Number of times the letter appears")


class LetterCounter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(CountLetters)

    def forward(self, word, letter):
        return self.predict(word=word, letter=letter)


def letter_counting_metric(example, prediction, trace=None) -> float:
    """
    Metric for letter counting task (CharBench format).
    Returns 1.0 for exact match, 0.0 otherwise.
    """
    try:
        predicted_count = int(prediction.count)
        correct_count = int(example.answer)
        return 1.0 if predicted_count == correct_count else 0.0

    except (ValueError, AttributeError, TypeError):
        # If we can't parse the response, it's wrong
        return 0.0


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for
        model: Model name for tokenizer selection

    Returns:
        Token count
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def log_instruction_sizes(program: dspy.Module, label: str = "Program"):
    """Log character and token counts for all instructions in a program."""
    print(f"\n--- {label} Instruction Sizes ---")
    total_chars = 0
    total_tokens = 0

    for name, pred in program.named_predictors():
        instruction = pred.signature.instructions
        chars = len(instruction)
        tokens = count_tokens(instruction)

        print(f"  {name}:")
        print(f"    Characters: {chars:,}")
        print(f"    Tokens (est): {tokens:,}")

        total_chars += chars
        total_tokens += tokens

    print("\n  Total:")
    print(f"    Characters: {total_chars:,}")
    print(f"    Tokens (est): {total_tokens:,}")
    print()


def log_compression_comparison(original: dspy.Module, compressed: dspy.Module, label: str = "Compression Results"):
    """Compare instruction sizes between original and compressed programs."""
    print(f"\n--- {label} ---")

    original_predictors = dict(original.named_predictors())
    compressed_predictors = dict(compressed.named_predictors())

    total_orig_chars = 0
    total_comp_chars = 0
    total_orig_tokens = 0
    total_comp_tokens = 0

    for name in original_predictors:
        orig_inst = original_predictors[name].signature.instructions
        comp_inst = compressed_predictors[name].signature.instructions

        orig_chars = len(orig_inst)
        comp_chars = len(comp_inst)
        orig_tokens = count_tokens(orig_inst)
        comp_tokens = count_tokens(comp_inst)

        char_ratio = comp_chars / orig_chars if orig_chars > 0 else 0
        token_ratio = comp_tokens / orig_tokens if orig_tokens > 0 else 0

        print(f"\n  {name}:")
        print(
            f"    Characters:  {orig_chars:,} → {comp_chars:,} "
            f"({char_ratio:.1%} retained, {(1 - char_ratio) * 100:.1f}% reduced)"
        )
        print(
            f"    Tokens (est): {orig_tokens:,} → {comp_tokens:,} "
            f"({token_ratio:.1%} retained, {(1 - token_ratio) * 100:.1f}% reduced)"
        )

        total_orig_chars += orig_chars
        total_comp_chars += comp_chars
        total_orig_tokens += orig_tokens
        total_comp_tokens += comp_tokens

    overall_char_ratio = total_comp_chars / total_orig_chars if total_orig_chars > 0 else 0
    overall_token_ratio = total_comp_tokens / total_orig_tokens if total_orig_tokens > 0 else 0

    print("\n  Overall:")
    print(f"    Total chars:   {total_orig_chars:,} → {total_comp_chars:,} ({overall_char_ratio:.1%})")
    print(f"    Total tokens (est): {total_orig_tokens:,} → {total_comp_tokens:,} ({overall_token_ratio:.1%})")
    print(f"    Char reduction: {(1 - overall_char_ratio) * 100:.1f}%")
    print(f"    Token reduction: {(1 - overall_token_ratio) * 100:.1f}%")
    print()


def init_dataset(
    train_size=100, val_size=50, test_size=200
) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    "Load CharBench dataset and prepare for DSPy."

    print("Loading CharBench dataset...")
    # Load and filter CharBench for character frequency counting
    # Using a specific revision or trusting the latest
    dataset = datasets.load_dataset("omriuz/CharBench")

    # Filter for character frequency counting task
    char_count = dataset["train"].filter(lambda x: x["task"] == "count_character_frequency")

    print(f"Dataset loaded: {len(char_count)} character counting examples")

    # Split into train/val/test
    # Ensure we don't go out of bounds
    total_needed = train_size + val_size + test_size
    if len(char_count) < total_needed:
        print(f"Warning: Requested {total_needed} examples but only {len(char_count)} available. Adjusting sizes.")
        # Simple adjustment logic could be added here, but for now let's assume enough data

    train_data = char_count.select(range(train_size))
    val_data = char_count.select(range(train_size, train_size + val_size))
    test_data = char_count.select(range(train_size + val_size, train_size + val_size + test_size))

    print(f"Dataset splits:\n  Train: {len(train_data)}\n  Val: {len(val_data)}\n  Test: {len(test_data)}")

    # Convert to DSPy Examples
    def to_dspy_examples(hf_dataset):
        return [
            dspy.Example(word=ex["word"], letter=ex["character"], answer=ex["answer"]).with_inputs("word", "letter")
            for ex in hf_dataset
        ]

    trainset = to_dspy_examples(train_data)
    valset = to_dspy_examples(val_data)
    testset = to_dspy_examples(test_data)

    return trainset, valset, testset


def evaluate_program(program, testset, name="Program"):
    """Evaluate a DSPy program on the test set."""
    print(f"\n--- Evaluating {name} ---")

    evaluator = dspy.Evaluate(
        devset=testset, metric=letter_counting_metric, num_threads=8, display_progress=True, display_table=0
    )

    result = evaluator(program)
    score = result.score
    print(f"{name} Accuracy: {score:.1f}%")
    return score


def run_example(ai_provider="localai"):
    """Run the full example with MIPROv2 optimization and CUTIA compression."""
    print("=== Letter Counting: MIPROv2 + CUTIA Pipeline ===")
    print(f"Using AI Provider: {ai_provider}")

    if ai_provider == "localai":
        LOCALAI_BASE_URL = os.getenv("LOCALAI_BASE_URL")
        LOCALAI_API_KEY = os.getenv("LOCALAI_API_KEY")

        if not LOCALAI_BASE_URL or not LOCALAI_API_KEY:
            raise ValueError("LOCALAI_BASE_URL and LOCALAI_API_KEY must be set for the localai provider")

        prompt_model = dspy.LM(
            model="openai/openai/gpt-oss-20b",
            api_base=LOCALAI_BASE_URL,
            api_key=LOCALAI_API_KEY,
            max_tokens=10000,
            temperature=1,
            cache=False,
            extra_body={"top_k": 40, "reasoning_effort": "medium"},
        )

        task_model = dspy.LM(
            model="openai/Qwen/Qwen3-0.6B",
            api_base=LOCALAI_BASE_URL,
            api_key=LOCALAI_API_KEY,
            max_tokens=2000,
            temperature=0.6,
            cache=False,
            extra_body={
                "min_p": 0,
                "top_p": 0.8,
                "top_k": 20,
                "presence_penalty": 1.5,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    elif ai_provider == "openai":
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set for the openai provider")

        prompt_model = dspy.LM(
            model="openai/gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            max_tokens=10000,
            temperature=1,
        )
        task_model = dspy.LM(
            model="openai/gpt-4.1-nano",
            api_key=OPENAI_API_KEY,
            max_tokens=2000,
            temperature=1,
        )
    elif ai_provider == "openrouter":
        OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

        if not OPENROUTER_BASE_URL or not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_BASE_URL and OPENROUTER_API_KEY must be set for the openrouter provider")

        prompt_model = dspy.LM(
            model="openrouter/openai/gpt-4o-mini",
            api_base=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            max_tokens=10000,
            temperature=1,
            cache=False,
        )
        task_model = dspy.LM(
            model="openrouter/openai/gpt-4.1-nano",
            api_base=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            max_tokens=2000,
            temperature=1,
            cache=False,
        )
    else:
        raise ValueError(f"Unknown AI provider: {ai_provider}")

    dspy.settings.configure(
        lm=task_model,
        adapter=dspy.JSONAdapter(),
        # enable_disk_cache=False,
        # enable_memory_cache=False,
    )

    # Load data
    try:
        trainset, valset, testset = init_dataset()
    except ImportError as e:
        print(e)
        return

    # 1. Baseline Evaluation
    print("\n" + "=" * 60)
    print("STAGE 1: Baseline Evaluation")
    print("=" * 60)

    student = LetterCounter()
    log_instruction_sizes(student, "Baseline Program")

    baseline_score = evaluate_program(student, testset, name="Baseline")

    # 2. MIPROv2 Optimization
    print("\n" + "=" * 60)
    print("STAGE 2: MIPROv2 Optimization")
    print("=" * 60)
    print("Optimizing with MIPROv2...")
    print("This may take several minutes...")
    print()

    mipro_optimizer = dspy.MIPROv2(
        metric=letter_counting_metric,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        auto="medium",
        prompt_model=prompt_model,
    )

    optimized_program = mipro_optimizer.compile(
        student,
        trainset=trainset,
        valset=valset,
        minibatch=False,
    )

    log_instruction_sizes(optimized_program, "Optimized Program")
    log_compression_comparison(student, optimized_program, "MIPROv2 Optimization Impact")

    optimized_score = evaluate_program(optimized_program, testset, name="Optimized (MIPROv2)")

    # 3. CUTIA Compression
    print("\n" + "=" * 60)
    print("STAGE 3: CUTIA Compression")
    print("=" * 60)
    print("Compressing optimized program with CUTIA...")
    print()

    optimizer = CUTIA(
        prompt_model=prompt_model,
        task_model=task_model,
        metric=letter_counting_metric,
        num_candidates=4,
        verbose=True,
    )

    compressed_program = optimizer.compile(
        student=optimized_program,
        trainset=trainset,
        valset=valset,
    )

    log_instruction_sizes(compressed_program, "Compressed Program")
    log_compression_comparison(optimized_program, compressed_program, "CUTIA Compression Impact")

    # 4. Evaluate Compressed Program
    compressed_score = evaluate_program(compressed_program, testset, name="Compressed (CUTIA)")

    # 5. Final Results Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    print("\nAccuracy Scores:")
    print(f"  Baseline:              {baseline_score:.1f}%")
    print(f"  After MIPROv2:         {optimized_score:.1f}% ({optimized_score - baseline_score:+.1f}%)")
    print(f"  After CUTIA:           {compressed_score:.1f}% ({compressed_score - optimized_score:+.1f}%)")
    print(f"  Overall Change:        {compressed_score - baseline_score:+.1f}%")

    # Overall compression from baseline to final
    log_compression_comparison(student, compressed_program, "Overall Pipeline Impact (Baseline → Final)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CharBench example with CUTIA.")
    parser.add_argument(
        "--ai-provider",
        type=str,
        default="localai",
        choices=["localai", "openai", "openrouter"],
        help="AI provider to use (default: localai)",
    )
    args = parser.parse_args()
    run_example(ai_provider=args.ai_provider)
