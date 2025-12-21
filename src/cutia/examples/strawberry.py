import argparse
import json
import os
import time
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


def create_experiment_directory(base_name: str = "strawberry", programs_dir: Path | None = None) -> Path:
    """
    Create timestamped experiment directory.

    Args:
        base_name: Base name for the experiment directory (default: "strawberry")
        programs_dir: Base programs directory (default: examples/programs/)

    Returns:
        Path to the created experiment directory
    """
    if programs_dir is None:
        programs_dir = Path(__file__).parent / "programs"

    programs_dir.mkdir(exist_ok=True)

    timestamp = int(time.time())
    experiment_dir = programs_dir / f"{base_name}_{timestamp}"
    experiment_dir.mkdir(exist_ok=True)

    print(f"Created experiment directory: {experiment_dir}")
    return experiment_dir


def save_experiment_metadata(experiment_dir: Path, config: dict, results: dict):
    """
    Save experiment configuration and results.

    Args:
        experiment_dir: Directory to save metadata in
        config: Experiment configuration dictionary
        results: Experiment results dictionary
    """
    metadata = {
        "config": config,
        "results": results,
        "timestamp": int(time.time()),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    metadata_path = experiment_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Saved metadata to: {metadata_path}")


def save_sample_llm_call(lm, experiment_dir: Path) -> str | None:
    """
    Save most recent LM call for debugging.

    Args:
        lm: The language model object with history
        experiment_dir: Directory to save the sample in

    Returns:
        Path where the sample was saved, or None if no history available
    """
    if not hasattr(lm, "history") or not lm.history:
        return None

    latest = lm.history[-1]
    call_data = {
        "timestamp": latest.get("timestamp"),
        "model": latest.get("model"),
        "prompt": latest.get("prompt"),
        "messages": latest.get("messages"),
        "outputs": latest.get("outputs"),
        "usage": latest.get("usage"),
        "cost": latest.get("cost"),
    }

    call_path = experiment_dir / "sample_llm_call.json"
    with open(call_path, "w") as f:
        json.dump(call_data, f, indent=2, default=str)

    return str(call_path)


def calculate_cost(lm) -> float:
    """
    Calculate total cost from LM history.

    Args:
        lm: The language model object with history

    Returns:
        Total cost in USD
    """
    if not hasattr(lm, "history") or not lm.history:
        return 0.0
    return sum([x.get("cost", 0) for x in lm.history if x.get("cost")])


def extract_token_stats(lm) -> dict:
    """
    Extract token usage statistics.

    Args:
        lm: The language model object with history

    Returns:
        Dictionary with token usage statistics
    """
    if not hasattr(lm, "history") or not lm.history:
        return {
            "total_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
        }

    total_prompt = sum(x.get("usage", {}).get("prompt_tokens", 0) for x in lm.history)
    total_completion = sum(x.get("usage", {}).get("completion_tokens", 0) for x in lm.history)

    return {
        "total_calls": len(lm.history),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
    }


def load_program_from_path(relative_path: str, stage: str, programs_dir: Path) -> Path:
    """
    Load a program from the programs directory.

    Args:
        relative_path: Directory name like 'strawberry_1703001234'
        stage: Which program to load ('optimized' or 'compressed')
        programs_dir: Base programs directory

    Returns:
        Path to program file (optimized_program.json or compressed_program.json)

    Raises:
        FileNotFoundError: If program not found
        ValueError: If invalid stage specified
    """
    if stage not in ["optimized", "compressed"]:
        raise ValueError(f"Invalid stage '{stage}'. Must be 'optimized' or 'compressed'")

    program_dir = programs_dir / relative_path
    program_file = program_dir / f"{stage}_program.json"

    if not program_dir.exists():
        raise FileNotFoundError(f"Program directory not found: {program_dir}")
    if not program_file.exists():
        raise FileNotFoundError(
            f"Program file not found: {program_file}\nAvailable files: {list(program_dir.glob('*.json'))}"
        )

    return program_file


def run_example(
    ai_provider: str = "localai",
    save_programs: bool = True,
    load_program: str | None = None,
    load_stage: str = "optimized",
    skip_optimization: bool = False,
    skip_compression: bool = False,
    output_dir: str | None = None,
):
    """
    Run the full example with MIPROv2 optimization and CUTIA compression.

    Args:
        ai_provider: AI provider to use (localai, openai, or openrouter)
        save_programs: Whether to save programs and metadata (default: True)
        load_program: Load program from programs/ directory (e.g., 'strawberry_1703001234')
        load_stage: Which program to load - 'optimized' or 'compressed' (default: 'optimized')
        skip_optimization: Skip MIPROv2 optimization stage (default: False)
        skip_compression: Skip CUTIA compression stage (default: False)
        output_dir: Custom output directory for programs (default: examples/programs/)
    """
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

    # Setup programs directory
    programs_dir = Path(__file__).parent / "programs" if output_dir is None else Path(output_dir)
    experiment_dir = None

    # Load data
    try:
        trainset, valset, testset = init_dataset()
    except ImportError as e:
        print(e)
        return

    # Initialize program
    student = LetterCounter()

    # Create experiment directory if saving (and not just evaluating)
    if save_programs and not (load_program and skip_optimization and skip_compression):
        experiment_dir = create_experiment_directory("strawberry", programs_dir)
        print(f"Results will be saved to: {experiment_dir}\n")

    # Load program if requested
    if load_program:
        program_path = load_program_from_path(load_program, load_stage, programs_dir)
        print(f"Loading {load_stage} program from: {program_path}")
        student.load(str(program_path))
        print("✓ Program loaded successfully\n")

        # Load and display metadata
        metadata_path = program_path.parent / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            print("Previous run info:")
            results = metadata.get("results", {})
            print(f"  Baseline: {results.get('baseline_score', 'N/A'):.1f}%")
            if "optimized_score" in results:
                print(f"  Optimized: {results.get('optimized_score', 'N/A'):.1f}%")
            if "compressed_score" in results:
                print(f"  Compressed: {results.get('compressed_score', 'N/A'):.1f}%")
            print()

    # 1. Baseline Evaluation
    print("=" * 60)
    print("STAGE 1: Baseline Evaluation")
    print("=" * 60)

    log_instruction_sizes(student, "Baseline Program")
    baseline_score = evaluate_program(student, testset, name="Baseline")

    # Track stages and results
    stages_completed = ["baseline"]
    results = {
        "baseline_score": baseline_score,
    }

    # 2. MIPROv2 Optimization
    optimized_program = student
    optimized_score = baseline_score

    if not skip_optimization:
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
        stages_completed.append("optimized")
        results["optimized_score"] = optimized_score
        results["optimization_gain"] = optimized_score - baseline_score

        # Save optimized program
        if save_programs and experiment_dir:
            optimized_program.save(str(experiment_dir / "optimized_program.json"))
            print(f"✓ Saved optimized program to {experiment_dir / 'optimized_program.json'}")
    else:
        print("\n" + "=" * 60)
        print("STAGE 2: Skipping MIPROv2 Optimization")
        print("=" * 60)

    # 3. CUTIA Compression
    compressed_program = optimized_program
    compressed_score = optimized_score

    if not skip_compression:
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
        )

        compressed_program = optimizer.compile(
            student=optimized_program,
            trainset=trainset,
            valset=valset,
        )

        log_instruction_sizes(compressed_program, "Compressed Program")
        log_compression_comparison(optimized_program, compressed_program, "CUTIA Compression Impact")

        compressed_score = evaluate_program(compressed_program, testset, name="Compressed (CUTIA)")
        stages_completed.append("compressed")
        results["compressed_score"] = compressed_score
        results["compression_loss"] = compressed_score - optimized_score
        results["overall_gain"] = compressed_score - baseline_score

        # Include compression stats
        if hasattr(compressed_program, "compression_stats"):
            results["compression_stats"] = compressed_program.compression_stats

        # Save compressed program
        if save_programs and experiment_dir:
            compressed_program.save(str(experiment_dir / "compressed_program.json"))
            print(f"✓ Saved compressed program to {experiment_dir / 'compressed_program.json'}")
    else:
        print("\n" + "=" * 60)
        print("STAGE 3: Skipping CUTIA Compression")
        print("=" * 60)

    # 4. Final Results and Metadata
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    cost = calculate_cost(task_model)
    token_stats = extract_token_stats(task_model)

    print("\nAccuracy Scores:")
    print(f"  Baseline:    {baseline_score:.1f}%")
    if "optimized_score" in results:
        print(f"  Optimized:   {optimized_score:.1f}% ({results['optimization_gain']:+.1f}%)")
    if "compressed_score" in results:
        print(f"  Compressed:  {compressed_score:.1f}% ({results['compression_loss']:+.1f}%)")
        print(f"  Overall:     {results['overall_gain']:+.1f}%")

    print(f"\nTotal cost: ${cost:.4f}")
    print(f"Total calls: {token_stats['total_calls']}")
    print(f"Total tokens: {token_stats['total_tokens']:,}")

    # Save metadata and sample call
    if save_programs and experiment_dir:
        results["cost"] = {
            "total_cost": cost,
            **token_stats,
        }

        config = {
            "experiment_name": "strawberry",
            "ai_provider": ai_provider,
            "task_model": task_model.model if hasattr(task_model, "model") else str(task_model),
            "prompt_model": prompt_model.model if hasattr(prompt_model, "model") else str(prompt_model),
            "train_size": len(trainset),
            "val_size": len(valset),
            "test_size": len(testset),
            "stages_completed": stages_completed,
            "mipro_config": {
                "max_bootstrapped_demos": 0,
                "max_labeled_demos": 0,
                "auto": "medium",
            },
            "cutia_config": {
                "num_candidates": 4,
            },
        }

        save_experiment_metadata(experiment_dir, config, results)
        save_sample_llm_call(task_model, experiment_dir)
        print(f"\n✓ All results saved to: {experiment_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CharBench letter counting with CUTIA compression.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: run full pipeline and save
  python strawberry.py

  # Skip optimization, compress baseline
  python strawberry.py --skip-optimization

  # Load previously optimized program and compress it
  python strawberry.py --load-program strawberry_1703001234 --load-stage optimized --skip-optimization

  # Load compressed program and evaluate
  python strawberry.py --load-program strawberry_1703001234 --load-stage compressed --skip-optimization --skip-compression

  # Run without saving
  python strawberry.py --no-save
        """,
    )
    parser.add_argument(
        "--ai-provider",
        type=str,
        default="localai",
        choices=["localai", "openai", "openrouter"],
        help="AI provider to use (default: localai)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save programs and metadata",
    )
    parser.add_argument(
        "--load-program",
        type=str,
        default=None,
        help="Load program from programs/ directory (e.g., 'strawberry_1703001234')",
    )
    parser.add_argument(
        "--load-stage",
        type=str,
        default="optimized",
        choices=["optimized", "compressed"],
        help="Which program to load: 'optimized' or 'compressed' (default: optimized)",
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip MIPROv2 optimization stage",
    )
    parser.add_argument(
        "--skip-compression",
        action="store_true",
        help="Skip CUTIA compression stage",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for programs (default: examples/programs/)",
    )

    args = parser.parse_args()

    run_example(
        ai_provider=args.ai_provider,
        save_programs=not args.no_save,
        load_program=args.load_program,
        load_stage=args.load_stage,
        skip_optimization=args.skip_optimization,
        skip_compression=args.skip_compression,
        output_dir=args.output_dir,
    )
