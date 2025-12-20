"""
CUTIA - Tree-Structured Evaluate Cut-Then-Transform Compressor

This module implements the CUTIA (Tree-based Recursive Evaluate Cut-Then-Transform)
method for prompt compression. It builds a segment tree of the prompt and performs
cut-or-rewrite decisions on each node.
"""

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.utils import get_signature, set_signature
from dspy.utils.parallelizer import ParallelExecutor

from .bounded_chat_adapter import BoundedChatAdapter

logger = logging.getLogger(__name__)

YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
ENDC = "\033[0m"

# Quality mode type definition (following MIPROv2 auto parameter pattern)
QualityModeType = Literal["strict", "balanced", "aggressive"]

# Mode-specific threshold deltas (in percentage points)
MODE_THRESHOLDS = {
    "strict": 0.0,  # No score degradation allowed
    # Use case: Safety-critical prompts, zero quality loss tolerance
    # Expected compression: 10-20%
    "balanced": -5.0,  # Moderate score degradation allowed (up to 5% drop)
    # Use case: Most applications, good quality/compression balance
    # Expected compression: 25-40%
    "aggressive": -10.0,  # Larger score degradation allowed (up to 10% drop)
    # Use case: Maximum compression priority, quality less critical
    # Expected compression: 40-60%
}


@dataclass
class RewriteCandidate:
    """Pre-generated rewrite option for a chunk."""

    rewritten_text: str
    target_compression_ratio: float
    generation_seed: int | None = None


@dataclass
class SegmentNode:
    """
    Represents a node in the segment tree.
    """

    node_id: str
    depth: int
    text: str
    span_start: int = 0
    span_end: int = 0

    # Tree structure
    left_child: Optional["SegmentNode"] = None
    right_child: Optional["SegmentNode"] = None

    # Split info (if this node was split)
    left_text: str | None = None
    chunk_text: str | None = None
    right_text: str | None = None
    chunk_reason: str | None = None

    # Decision info
    status: str = "pending"  # pending, cut, rewritten, kept
    replacement_text: str | None = None
    score_before: float = 0.0
    score_after: float = 0.0
    saved_tokens: int = 0

    # New fields for pre-generated options
    rewrite_candidates: list[RewriteCandidate] | None = None
    split_attempts: int = 0  # Track retries during building

    def get_rendered_text(self) -> str:
        """Returns the text for this node based on its status and children."""
        if self.status == "cut":
            # If cut, we might still have children if the cut happened at this level's chunk
            # But wait, if we cut the *chunk*, what happens to left and right?
            # The PRD says: "Attempt cut by removing chunk_text and stitching left_text + right_text"
            # So if status is cut, we return left + right (potentially processed recursively)
            # Actually, the PRD says: "Traverses nodes in post-order... Parent reconstruction uses the resolved child outputs"

            # If this node is a leaf or we decided to cut/rewrite THIS node's focal chunk:
            # The node represents a span. The split gave us left, chunk, right.
            # If we cut 'chunk', we keep left and right.
            # But left and right might be complex nodes themselves.

            # Let's clarify:
            # A node represents a text span.
            # It is split into L, C, R.
            # L and R become children nodes (left_child, right_child).
            # C is the "focal chunk" at this level.

            l_text = self.left_child.get_rendered_text() if self.left_child else (self.left_text or "")
            r_text = self.right_child.get_rendered_text() if self.right_child else (self.right_text or "")

            return l_text + r_text

        if self.status == "rewritten":
            l_text = self.left_child.get_rendered_text() if self.left_child else (self.left_text or "")
            r_text = self.right_child.get_rendered_text() if self.right_child else (self.right_text or "")
            return l_text + (self.replacement_text or "") + r_text

        # kept or pending
        # If we have children, we must use them to reconstruct, because they might have cuts inside them
        if self.left_child or self.right_child:
            l_text = self.left_child.get_rendered_text() if self.left_child else (self.left_text or "")
            c_text = self.chunk_text or ""  # The chunk at this level is kept
            r_text = self.right_child.get_rendered_text() if self.right_child else (self.right_text or "")
            return l_text + c_text + r_text
        return self.text


class ProposeChunk(dspy.Signature):
    """
    Analize instraction_to_analyze that is used for calls to an LM, then identify a chunk within the instraction_to_analyze that has the most potential to safely being removed or rewritten with the goal to make instraction_to_analyze shorter while keeping the task of the instraction_to_analyze fully clear and complete.

    IMPORTANT RULES:
    1. If no valid chunk can be found, respond with has_chunk=false and leave left, chunk, right as null.
    2. If there is a valid chunk to consider for removal or rewrite - output exactly three parts: left, chunk, right; left + chunk + right must reconstruct the instraction_to_analyze EXACTLY.
    3. Do NOT add any formatting, markers, or explanatory text in left, chunk, right fields.
    4. Preserve all original whitespace and punctuation of the instraction_to_analyze.

    The chunk you select could be:
    - Redundant or overly verbose content
    - Examples that could be shortened
    - Repetitive phrases
    - Unnecessary explanations
    """

    instraction_to_analyze = dspy.InputField(desc="The instraction to analyze")

    has_chunk = dspy.OutputField(
        format=bool, desc="Is there a valid chunk that can be removed or rewritten from the instraction_to_analyze?"
    )
    left = dspy.OutputField(
        desc="The EXACT left part of the instraction_to_analyze that appears before the chunk (no modifications), nullable"
    )
    chunk = dspy.OutputField(
        desc="The EXACT chunk to potentially remove or rewrite from the instraction_to_analyze (no modifications), nullable"
    )
    right = dspy.OutputField(
        desc="The EXACT right part of the instraction_to_analyze that appears after the chunk (no modifications), nullable"
    )
    chunk_reason = dspy.OutputField(desc="Brief explanation (1-2 sentences) of why a chunk was selected or not")


class RewriteChunk(dspy.Signature):
    """
    Rewrite the text to be shorter while preserving its essential meaning.
    """

    text = dspy.InputField()
    target_length = dspy.InputField(desc="Approximate target length in characters")
    rewritten_text = dspy.OutputField()


class MultiVariantRewriteChunk(dspy.Signature):
    """
    Rewrite the text to be shorter while preserving its essential meaning.
    Generate two variants:
    1. A concise summary (high compression)
    2. A detailed summary (moderate compression, preserving more details)
    """

    text = dspy.InputField()
    concise_summary = dspy.OutputField(desc="High compression summary")
    detailed_summary = dspy.OutputField(desc="Moderate compression summary with key details")


class ValidateReconstruction(dspy.Signature):
    """
    Verify that the reconstructed text is semantically equivalent to the original text.

    Check if the meaning, structure, and content are preserved. Minor formatting differences
    (whitespace, punctuation) are acceptable, but the semantic content must match.
    """

    original_text = dspy.InputField(desc="The original text")
    reconstructed_text = dspy.InputField(desc="The reconstructed text (left + chunk + right)")

    reasoning = dspy.OutputField(desc="Brief explanation of whether they match semantically")
    is_valid = dspy.OutputField(desc="'yes' if semantically equivalent, 'no' if different")


class CUTIA(Teleprompter):
    """CUTIA - Tree-Structured Evaluate Cut-Then-Transform Compressor

    Multi-candidate prompt compression using tree-based segmentation and node-level
    cut/rewrite decisions.

    Key Features:
    - Trainset minibatch exploration: Each candidate explores different trainset subsamples
      for node decisions (following bootstrap aggregating theory)
    - Valset validation: All candidates evaluated on full valset for unbiased selection
    - Quality-first selection: Balances compression ratio with quality thresholds

    See docs/trainset-minibatch-exploration-analysis.md for theoretical foundation.

    Args:
        node_decision_minibatch: Enable minibatch sampling for node decisions (default: True)
        node_decision_minibatch_size: Minibatch size (default: None = valset_size)
                                      Each candidate samples this many examples from trainset
                                      for node decisions. Default equals valset size to provide
                                      comparable sample sizes for exploration and validation.

        top_k_candidates: After generating all candidates, take the top-K candidates (according
                 to `candidate_selection` using VALSET scores), then re-evaluate those
                 K candidates on the full TRAINSET and return the best-by-trainset.
                 (default: 2)
    """

    def __init__(
        self,
        prompt_model=None,
        task_model=None,
        metric=None,
        max_depth: int = 5,
        min_chunk_chars: int = 50,
        quality_mode: QualityModeType = "strict",
        prompt_retries: int = 2,
        target_compression_ratio: float = 0.5,
        track_stats: bool = True,
        num_threads: int = 8,
        num_candidates: int = 4,
        candidate_seed_offset: int = 1000,
        seed: int = 42,
        candidate_selection: str = "weighted",  # "quality_first"
        quality_weight: float = 0.3,
        compression_weight: float = 0.7,
        node_decision_minibatch: bool = True,
        node_decision_minibatch_size: int | None = None,
        top_k_candidates: int = 2,
        traversal_strategy: str = "pre_order",  # "post_order", "pre_order", "random"
        parallel_tree_building: bool = True,
        tree_building_threads: int | None = None,
        enable_cutting: bool = True,
        rewrite_strategy: str = "basic",  # "basic", "multi_variant"
    ):
        self.prompt_model = prompt_model or dspy.settings.lm
        self.task_model = task_model or dspy.settings.lm
        self.metric = metric
        self.max_depth = max_depth
        self.min_chunk_chars = min_chunk_chars
        self.traversal_strategy = traversal_strategy
        self.parallel_tree_building = parallel_tree_building
        self.tree_building_threads = tree_building_threads or num_threads
        self.enable_cutting = enable_cutting
        self.rewrite_strategy = rewrite_strategy
        self.quality_mode = quality_mode
        self.prompt_retries = prompt_retries
        self.target_compression_ratio = target_compression_ratio
        self.track_stats = track_stats
        self.num_threads = num_threads
        self.num_candidates = num_candidates
        self.candidate_seed_offset = candidate_seed_offset
        self.seed = seed
        self.candidate_selection = candidate_selection
        self.quality_weight = quality_weight
        self.compression_weight = compression_weight
        self.node_decision_minibatch = node_decision_minibatch
        self.node_decision_minibatch_size = node_decision_minibatch_size

        if top_k_candidates < 1:
            raise ValueError("top_k_candidates must be >= 1")
        self.top_k_candidates = int(top_k_candidates)

        self.stats = {"nodes_visited": 0, "nodes_cut": 0, "nodes_rewritten": 0, "llm_calls": 0}

    def _create_node_decision_minibatch(
        self,
        trainset: list[dspy.Example],
        valset_size: int,
        candidate_seed: int,
    ) -> list[dspy.Example]:
        """Create a random minibatch from trainset for node decision making.

        Following the analysis in docs/trainset-minibatch-exploration-analysis.md,
        this method uses trainset for exploration (node decisions) while valset
        is reserved for final candidate selection. This aligns with bootstrap
        aggregating theory and proper train/val separation.

        Args:
            trainset: Training set to sample from (exploration data)
            valset_size: Size of validation set (used for default minibatch size)
            candidate_seed: Seed for this candidate (ensures different minibatch per candidate)

        Returns:
            Random subset of trainset for node decision evaluation
        """
        import random

        # TODO: it should be valset by default?
        if not self.node_decision_minibatch:
            return trainset  # Use full trainset if minibatch disabled

        # Determine minibatch size
        if self.node_decision_minibatch_size is not None:
            minibatch_size = self.node_decision_minibatch_size
        else:
            # Default: same size as valset (per user request)
            # This allows exploration on trainset with similar sample size to final evaluation
            minibatch_size = max(1, valset_size)

        # Cap at trainset size
        minibatch_size = min(minibatch_size, len(trainset))

        # Create deterministic random sample using candidate seed
        rng = random.Random(candidate_seed)
        indices = rng.sample(range(len(trainset)), minibatch_size)

        # Return minibatch in original order (sorted indices)
        minibatch = [trainset[i] for i in sorted(indices)]

        logger.info(
            f"Created node decision minibatch from TRAINSET: {len(minibatch)}/{len(trainset)} examples "
            f"(target size: {valset_size}, seed: {candidate_seed})"
        )

        return minibatch

    def compile(
        self,
        student: dspy.Module,
        *,
        trainset: list[dspy.Example],
        valset: list[dspy.Example] | None = None,
        minibatch: bool = True,
        minibatch_size: int = 50,
        **kwargs,
    ) -> dspy.Module:
        self.student = student

        # Setup validation set (Match CUTO logic)
        if valset is None:
            valset_size = min(100, max(1, int(len(trainset) * 0.2)))
            cutoff = len(trainset) - valset_size
            valset = trainset[cutoff:]
            trainset = trainset[:cutoff]
            logger.info(f"Auto-split validation set: {len(valset)} examples")

        self.trainset = trainset
        self.valset = valset

        # Determine evaluation set (minibatch or full)
        if minibatch and minibatch_size < len(valset):
            eval_set = valset[:minibatch_size]
            logger.info(f"Using minibatch evaluation: {minibatch_size} examples")
        else:
            eval_set = valset
            logger.info(f"Using full validation set: {len(eval_set)} examples")

        # Calculate global baseline on VALSET for final selection
        # This ensures we compare final candidates against a consistent standard
        logger.info("=" * 60)
        logger.info("BASELINE EVALUATION")
        logger.info("=" * 60)
        baseline_program = student.deepcopy()
        global_baseline = self._evaluate(baseline_program, valset)
        logger.info(f"Global Baseline on VALSET: {BOLD}{global_baseline:.1f}%{ENDC}")

        # Multi-Candidate Generation and Selection
        logger.info(f"\n{'=' * 60}")
        logger.info("CANDIDATE GENERATION")
        logger.info("=" * 60)
        logger.info(f"Generating {self.num_candidates} compression candidates...")
        logger.info("Each candidate uses its own baseline calculated on its trainset minibatch")
        candidates = []

        for i in range(self.num_candidates):
            candidate_seed = self.seed + i * self.candidate_seed_offset

            # Create minibatch from TRAINSET for this candidate's node decisions
            # Following trainset-minibatch-exploration-analysis.md:
            # - Trainset used for exploration (node decisions)
            # - Valset reserved for final candidate selection
            # This aligns with bootstrap aggregating and proper train/val separation
            node_decision_set = self._create_node_decision_minibatch(
                trainset=trainset, valset_size=len(valset), candidate_seed=candidate_seed
            )

            # Calculate candidate-specific baseline on SAME minibatch
            # This ensures fair comparison: baseline and node scores from same dataset
            baseline_program = student.deepcopy()
            candidate_baseline = self._evaluate(baseline_program, node_decision_set)

            # Determine strategy for this candidate
            candidate_strategy = self.traversal_strategy
            if candidate_strategy == "random":
                import random

                candidate_strategy = random.choice(["post_order", "pre_order"])

            logger.info(f"\n{'=' * 60}")
            logger.info(f"{BOLD}Candidate {i + 1}/{self.num_candidates}{ENDC} (seed: {candidate_seed})")
            logger.info(f"  Strategy: {BLUE}{candidate_strategy}{ENDC}")
            logger.info(
                f"  Baseline: {BOLD}{candidate_baseline:.1f}%{ENDC} (on {len(node_decision_set)} TRAINSET examples)"
            )
            logger.info(f"  Threshold: {YELLOW}{candidate_baseline + MODE_THRESHOLDS[self.quality_mode]:.1f}%{ENDC}")
            logger.info(f"  Node decisions: {len(node_decision_set)} examples from TRAINSET")
            logger.info(f"  Final evaluation: {len(valset)} examples from VALSET")
            logger.info(f"{'=' * 60}")

            # Compress with specific seed and candidate-specific baseline
            compressed = self._compress_with_seed(
                student,
                trainset,
                valset,
                node_decision_set,
                candidate_baseline,
                candidate_seed,
                strategy=candidate_strategy,
            )

            # Evaluate candidate on FULL valset for selection
            score = self._evaluate(compressed, valset)

            # Get compression metrics
            compression_ratio = self._calculate_compression_ratio(compressed)
            original_tokens, compressed_tokens = self._get_token_counts(compressed)

            candidates.append(
                {
                    "program": compressed,
                    "score": score,
                    "seed": candidate_seed,
                    "compression_ratio": compression_ratio,
                    "original_tokens": original_tokens,
                    "compressed_tokens": compressed_tokens,
                    "node_decision_set_size": len(node_decision_set),
                    "final_eval_set_size": len(valset),
                    "candidate_baseline": candidate_baseline,
                    "strategy": candidate_strategy,
                }
            )

            logger.info(
                f"\nCandidate {i + 1}/{self.num_candidates}: "
                f"Score: {GREEN}{BOLD}{score:.1f}%{ENDC} (on {len(valset)} VALSET examples), "
                f"Compression: {BLUE}{BOLD}{compression_ratio:.2f}{ENDC}, "
                f"Tokens: {compressed_tokens}/{original_tokens}, "
                f"Strategy: {candidate_strategy}, "
                f"Node decisions based on: {len(node_decision_set)} TRAINSET examples"
            )

        # Select best candidate using configured strategy
        logger.info(f"\n{'=' * 60}")
        logger.info("CANDIDATE SELECTION SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"Strategy: {BLUE}{self.candidate_selection}{ENDC}")
        logger.info(f"Global Baseline (VALSET): {BOLD}{global_baseline:.1f}%{ENDC}")
        logger.info("Candidate-specific baselines used for node decisions")
        logger.info(f"Quality mode: {YELLOW}{self.quality_mode}{ENDC}")
        logger.info(f"Threshold offset: {YELLOW}{MODE_THRESHOLDS[self.quality_mode]:+.1f}%{ENDC}")
        logger.info("\nCandidate Results:")
        for i, c in enumerate(candidates, 1):
            logger.info(
                f"  {i}. Baseline: {BOLD}{c['candidate_baseline']:.1f}%{ENDC}, "
                f"Score: {GREEN}{c['score']:.1f}%{ENDC}, "
                f"Compression: {BLUE}{c['compression_ratio']:.2f}{ENDC}, "
                f"Tokens: {c['compressed_tokens']}/{c['original_tokens']}, "
                f"Strategy: {c['strategy']}, "
                f"Seed: {c['seed']}"
            )

        top_candidates = self._select_best_candidate(candidates, global_baseline, score_key="score")
        top_k = len(top_candidates)

        logger.info(f"\nTop-{top_k} candidates by strategy ({BLUE}{self.candidate_selection}{ENDC}):")
        for rank, c in enumerate(top_candidates, 1):
            logger.info(
                f"  {rank}. Candidate #{candidates.index(c) + 1}: "
                f"VALSET Score: {GREEN}{c['score']:.1f}%{ENDC}, "
                f"Compression: {BLUE}{c['compression_ratio']:.2f}{ENDC}, "
                f"Strategy: {c['strategy']}, "
                f"Seed: {c['seed']}"
            )

        logger.info(f"\n{'=' * 60}")
        logger.info(f"TOP-{top_k} RE-EVALUATION ON TRAINSET")
        logger.info(f"{'=' * 60}")
        logger.info(f"Re-evaluating Top-{top_k} candidates on TRAINSET: {len(trainset)} examples")
        for c in top_candidates:
            c["trainset_score"] = self._evaluate(c["program"], trainset)
            logger.info(
                f"  Candidate #{candidates.index(c) + 1}: TRAINSET Score: {GREEN}{c['trainset_score']:.1f}%{ENDC} "
                f"(VALSET: {c['score']:.1f}%, Compression: {BLUE}{c['compression_ratio']:.2f}{ENDC})"
            )

        # Use the same selection strategy for the final decision, but based on TRAINSET scores
        # Note: For quality_first, we use global_baseline (from Valset) as a proxy threshold
        final_selection = self._select_best_candidate(top_candidates, global_baseline, score_key="trainset_score")
        best = final_selection[0]

        logger.info(f"\n{'=' * 60}")
        logger.info("FINAL SELECTION")
        logger.info(f"{'=' * 60}")
        logger.info(f"{GREEN}{BOLD}Selected Candidate (best-of-best): #{candidates.index(best) + 1}{ENDC}")
        logger.info(f"  TRAINSET Score: {GREEN}{BOLD}{best.get('trainset_score', 0.0):.1f}%{ENDC}")
        logger.info(f"  VALSET Score: {GREEN}{best['score']:.1f}%{ENDC}")
        logger.info(f"  Compression Ratio: {BLUE}{BOLD}{best['compression_ratio']:.2f}{ENDC}")
        logger.info(f"  Token Reduction: {BOLD}{best['original_tokens'] - best['compressed_tokens']}{ENDC} tokens")
        logger.info(f"{'=' * 60}")

        # Track all candidates if requested
        if self.track_stats:
            best["program"].compression_candidates = candidates

        return best["program"]

    def _compress_with_seed(
        self,
        student: dspy.Module,
        trainset: list[dspy.Example],
        valset: list[dspy.Example],
        node_decision_set: list[dspy.Example],
        baseline_score: float,
        seed: int,
        strategy: str = "post_order",
    ) -> dspy.Module:
        """
        Compress the student program using a specific random seed.
        This allows generating diverse compression candidates.

        Args:
            student: The DSPy program to compress
            trainset: Training dataset
            valset: Full validation set (used for final candidate selection)
            node_decision_set: Dataset to use for evaluating node-level decisions
                              (trainset minibatch by default, following bootstrap aggregating theory)
            baseline_score: Baseline score for quality threshold
            seed: Random seed for this candidate
            strategy: Tree traversal strategy ("post_order" or "pre_order")
        """
        import random

        # Set random seed for reproducibility of this candidate
        random.seed(seed)

        # Create deep copy to avoid modifying original
        compressed_program = student.deepcopy()

        all_stats = {}

        for i, predictor in enumerate(compressed_program.predictors()):
            signature = get_signature(predictor)
            original_instructions = signature.instructions

            if not original_instructions:
                continue

            logger.info(f"Compressing instructions for predictor {i}: {original_instructions[:50]}...")

            # Build Tree
            logger.info("Building segment tree...")

            if self.parallel_tree_building:
                root_node = self._build_tree_parallel(original_instructions, depth=0, node_id="root")
            else:
                root_node = self._build_tree(original_instructions, depth=0, node_id="root")

            logger.info("Tree built.")

            # Compress (Traverse and Optimize)
            self._process_node(
                root_node,
                root_node,
                predictor,
                compressed_program,
                baseline_score,
                node_decision_set,
                strategy=strategy,
            )

            # Reconstruct
            final_instructions = root_node.get_rendered_text()

            # Apply final instructions
            updated_signature = signature.with_instructions(final_instructions)
            set_signature(predictor, updated_signature)

            # Log stats
            logger.info(
                f"Final instructions length: {len(final_instructions)} (Original: {len(original_instructions)})"
            )

            all_stats[i] = {
                "original_length": len(original_instructions),
                "final_length": len(final_instructions),
                "nodes_visited": self.stats["nodes_visited"],
                "nodes_cut": self.stats["nodes_cut"],
                "nodes_rewritten": self.stats["nodes_rewritten"],
                "baseline_score": baseline_score,
                "final_score": self._evaluate(compressed_program, node_decision_set),
            }

            # Reset stats for next predictor
            self.stats = dict.fromkeys(self.stats, 0)

        if self.track_stats:
            compressed_program.compression_stats = self._build_stats(all_stats)

        return compressed_program

    def _build_stats(self, all_stats: dict) -> dict:
        """Build comprehensive compression statistics matching the expected format."""
        compression_stats = {
            "original_total_tokens": 0,
            "compressed_total_tokens": 0,
            "original_total_chars": 0,
            "compressed_total_chars": 0,
            "per_signature_stats": [],
            "mode": "CUTIA",
            "quality_mode": self.quality_mode,
        }

        for predictor_idx, sig_stats in all_stats.items():
            original_chars = sig_stats.get("original_length", 0)
            compressed_chars = sig_stats.get("final_length", 0)

            # Estimate tokens (approx 4 chars per token)
            original_tokens = max(1, int(original_chars / 4))
            compressed_tokens = max(1, int(compressed_chars / 4))

            compression_stats["original_total_chars"] += original_chars
            compression_stats["compressed_total_chars"] += compressed_chars
            compression_stats["original_total_tokens"] += original_tokens
            compression_stats["compressed_total_tokens"] += compressed_tokens

            compression_stats["per_signature_stats"].append(
                {
                    "predictor_index": predictor_idx,
                    "original_chars": original_chars,
                    "compressed_chars": compressed_chars,
                    "compression_ratio": compressed_chars / original_chars if original_chars > 0 else 1.0,
                    "original_tokens": original_tokens,
                    "compressed_tokens": compressed_tokens,
                    "token_compression_ratio": compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
                    "nodes_visited": sig_stats.get("nodes_visited", 0),
                    "nodes_cut": sig_stats.get("nodes_cut", 0),
                    "nodes_rewritten": sig_stats.get("nodes_rewritten", 0),
                    "baseline_score": sig_stats.get("baseline_score", 0.0),
                    "final_score": sig_stats.get("final_score", 0.0),
                }
            )

        # Calculate overall ratios
        if compression_stats["original_total_tokens"] > 0:
            compression_stats["token_compression_ratio"] = (
                compression_stats["compressed_total_tokens"] / compression_stats["original_total_tokens"]
            )
        else:
            compression_stats["token_compression_ratio"] = 1.0

        if compression_stats["original_total_chars"] > 0:
            compression_stats["compression_ratio"] = (
                compression_stats["compressed_total_chars"] / compression_stats["original_total_chars"]
            )
        else:
            compression_stats["compression_ratio"] = 1.0

        return compression_stats

    def _calculate_compression_ratio(self, program: dspy.Module) -> float:
        """Calculate compression ratio from program's compression_stats."""
        if hasattr(program, "compression_stats"):
            stats = program.compression_stats
            return stats.get("compression_ratio", 1.0)
        return 1.0  # No compression

    def _get_token_counts(self, program: dspy.Module) -> tuple[int, int]:
        """Get original and compressed token counts from program's compression_stats."""
        if hasattr(program, "compression_stats"):
            stats = program.compression_stats
            return (
                stats.get("original_total_tokens", 0),
                stats.get("compressed_total_tokens", 0),
            )
        return (0, 0)

    def _select_best_candidate(
        self, candidates: list[dict], global_baseline: float, score_key: str = "score"
    ) -> list[dict]:
        """Select top-K candidates based on configured strategy.

        Returns:
            A list of up to `self.top_k_candidates` candidate dicts, ordered best->worst.
        """
        k = min(self.top_k_candidates, len(candidates))
        if k <= 0:
            return []

        if self.candidate_selection == "quality_first":
            ranked = self._select_quality_first(candidates, global_baseline, score_key)
        elif self.candidate_selection == "weighted":
            ranked = self._select_weighted(candidates, global_baseline, score_key)
        elif self.candidate_selection == "best_score":
            ranked = self._select_best_score(candidates, score_key)
        else:
            raise ValueError(f"Unknown selection strategy: {self.candidate_selection}")

        return ranked[:k]

    def _select_quality_first(
        self, candidates: list[dict], global_baseline: float, score_key: str = "score"
    ) -> list[dict]:
        """Rank candidates by best compression among those meeting quality threshold.

        Each candidate is evaluated against the global baseline + threshold, ensuring fair
        comparison across candidates.
        """
        # Filter candidates meeting the global quality threshold
        acceptable = []
        for c in candidates:
            candidate_threshold = global_baseline + MODE_THRESHOLDS[self.quality_mode]
            if c.get(score_key, float("-inf")) >= candidate_threshold:
                acceptable.append(c)

        if not acceptable:
            # No candidates meet threshold, fall back to best score
            logger.warning("No candidates meet quality threshold. Falling back to best score ranking.")
            return sorted(
                candidates, key=lambda x: (x.get(score_key, float("-inf")), -x["compression_ratio"]), reverse=True
            )

        ranked = sorted(acceptable, key=lambda x: (x["compression_ratio"], -x.get(score_key, float("-inf"))))

        logger.info(
            f"Ranked {len(ranked)} acceptable candidates by compression (quality threshold met: "
            f"{global_baseline + MODE_THRESHOLDS[self.quality_mode]:.1f}%)"
        )

        return ranked

    def _select_weighted(self, candidates: list[dict], global_baseline: float, score_key: str = "score") -> list[dict]:
        """Rank candidates by weighted score combining quality and compression.

        Prioritizes candidates that meet the quality threshold.
        """
        # Normalize scores to 0-1 range
        scores = [c.get(score_key, 0.0) for c in candidates]
        min_score, max_score = min(scores), max(scores)
        score_range = max_score - min_score if max_score > min_score else 1.0

        ratios = [c["compression_ratio"] for c in candidates]
        min_ratio, max_ratio = min(ratios), max(ratios)
        ratio_range = max_ratio - min_ratio if max_ratio > min_ratio else 1.0

        # Calculate combined scores
        for c in candidates:
            score_val = c.get(score_key, 0.0)
            normalized_score = (score_val - min_score) / score_range if score_range > 0 else 1.0
            # Lower ratio is better, so invert: (max - current) / range
            normalized_compression = (max_ratio - c["compression_ratio"]) / ratio_range if ratio_range > 0 else 1.0

            c["combined_score"] = (
                self.quality_weight * normalized_score + self.compression_weight * normalized_compression
            )

        # Determine acceptability threshold
        threshold = global_baseline + MODE_THRESHOLDS[self.quality_mode]

        ranked = sorted(
            candidates,
            key=lambda x: (
                x.get(score_key, float("-inf")) >= threshold,  # Acceptable candidates first
                x.get("combined_score", float("-inf")),
                x.get(score_key, float("-inf")),
                -x["compression_ratio"],
            ),
            reverse=True,
        )

        if ranked:
            best = ranked[0]
            logger.info(
                f"Top candidate by combined score (acceptable first): {best['combined_score']:.3f} "
                f"({best.get(score_key, 0.0):.1f}% quality, {best['compression_ratio']:.2f} compression)"
            )

        return ranked

    def _select_best_score(self, candidates: list[dict], score_key: str = "score") -> list[dict]:
        """Rank candidates by best score (original behavior)."""
        return sorted(
            candidates, key=lambda x: (x.get(score_key, float("-inf")), -x["compression_ratio"]), reverse=True
        )

    def _strip_dspy_markers(self, text: str) -> str:
        """Remove DSPy field markers from text to prevent format instruction leakage."""
        import re

        # Remove markers like [[ ## fieldname ## ]]
        text = re.sub(r"\[\[\s*##\s*\w+\s*##\s*\]\]", "", text)
        # Remove format instructions about markers
        text = re.sub(r"Respond with.*?field.*?\[\[.*?\]\].*?\.", "", text, flags=re.IGNORECASE | re.DOTALL)
        return text.strip()

    def _clean_llm_string(self, val) -> str:
        """Clean string values from LLM, handling 'null' and None."""
        if val is None:
            return ""
        s = str(val)
        if s.lower().strip() == "null":
            return ""
        return s

    def _build_tree(self, text: str, depth: int, node_id: str) -> SegmentNode:
        logger.info(f"Building tree node {node_id} (Depth {depth}, Length {len(text)})")

        node = SegmentNode(node_id=node_id, depth=depth, text=text)

        if depth >= self.max_depth:
            logger.info(f"  Leaf: Max depth reached at {node_id}")
            return node

        if len(text) < self.min_chunk_chars:
            logger.info(f"  Leaf: Text too short at {node_id} ({len(text)} < {self.min_chunk_chars})")
            return node

        # Call LLM to propose split
        proposer = dspy.Predict(ProposeChunk)

        for attempt in range(self.prompt_retries + 1):
            try:
                self.stats["llm_calls"] += 1
                logger.info(f"  Requesting split for {node_id} (Attempt {attempt + 1})")

                # Use BoundedChatAdapter for clear input/output boundaries
                with dspy.settings.context(trace=[], lm=self.prompt_model, adapter=BoundedChatAdapter()):
                    pred = proposer(instraction_to_analyze=text)
                    logger.info(f"  Proposer prediction: {pred}")
                    # if hasattr(self.prompt_model, "history") and self.prompt_model.history:
                    #     logger.info(f"  Full Prompt: {self.prompt_model.history[-1]}")
                    #     exit()

                has_chunk = getattr(pred, "has_chunk", False)
                if isinstance(has_chunk, str):
                    has_chunk = has_chunk.lower() == "true"

                if not has_chunk:
                    logger.info(f"  No chunk proposed for {node_id}")
                    return node

                left = self._clean_llm_string(getattr(pred, "left", None))
                chunk = self._clean_llm_string(getattr(pred, "chunk", None))
                right = self._clean_llm_string(getattr(pred, "right", None))

                # Strip any leaked DSPy markers
                # left = self._strip_dspy_markers(left)
                # chunk = self._strip_dspy_markers(chunk)
                # right = self._strip_dspy_markers(right)

                logger.info(f"  Original prompt segment: {text}")
                logger.info(f"  Left: {left}")
                logger.info(f"  Chunk: {chunk}")
                logger.info(f"  Right: {right}")
                logger.info(f"  Proposed split: L={len(left)}, C={len(chunk)}, R={len(right)}")

                # Enable LLM-based reconstruction validation
                reconstructed = left + chunk + right

                if not self._validate_reconstruction(text, reconstructed):
                    logger.warning(f"  Reconstruction validation failed at {node_id}")
                    continue  # Retry with next attempt

                # If valid, populate node
                node.left_text = left
                node.chunk_text = chunk
                node.right_text = right
                node.chunk_reason = getattr(pred, "chunk_reason", None)

                logger.info(f"  Split accepted for {node_id}")

                # Recurse
                if left and len(left) >= self.min_chunk_chars:
                    node.left_child = self._build_tree(left, depth + 1, f"{node_id}.L")

                if right and len(right) >= self.min_chunk_chars:
                    node.right_child = self._build_tree(right, depth + 1, f"{node_id}.R")

                return node

            except Exception as e:
                logger.warning(f"Error building tree at {node_id}: {e}")

        logger.info(f"  Failed to split node {node_id} after retries")
        return node

    def _generate_split(self, node: SegmentNode) -> SegmentNode:
        """Attempt to split a node using LLM. Returns the node with split info populated."""
        if node.depth >= self.max_depth or len(node.text) < self.min_chunk_chars:
            return node

        # Call LLM to propose split
        proposer = dspy.Predict(ProposeChunk)

        for attempt in range(self.prompt_retries + 1):
            try:
                self.stats["llm_calls"] += 1
                # Use BoundedChatAdapter for clear input/output boundaries
                with dspy.settings.context(trace=[], lm=self.prompt_model, adapter=BoundedChatAdapter()):
                    pred = proposer(instraction_to_analyze=node.text)

                has_chunk = getattr(pred, "has_chunk", False)
                if isinstance(has_chunk, str):
                    has_chunk = has_chunk.lower() == "true"

                if not has_chunk:
                    return node

                left = self._clean_llm_string(getattr(pred, "left", None))
                chunk = self._clean_llm_string(getattr(pred, "chunk", None))
                right = self._clean_llm_string(getattr(pred, "right", None))

                # Enable LLM-based reconstruction validation
                reconstructed = left + chunk + right

                if not self._validate_reconstruction(node.text, reconstructed):
                    continue  # Retry with next attempt

                # If valid, populate node
                node.left_text = left
                node.chunk_text = chunk
                node.right_text = right
                node.chunk_reason = getattr(pred, "chunk_reason", None)

                # Create children placeholders (but don't recurse yet)
                if left and len(left) >= self.min_chunk_chars:
                    node.left_child = SegmentNode(node_id=f"{node.node_id}.L", depth=node.depth + 1, text=left)

                if right and len(right) >= self.min_chunk_chars:
                    node.right_child = SegmentNode(node_id=f"{node.node_id}.R", depth=node.depth + 1, text=right)

                return node

            except Exception as e:
                logger.warning(f"Error splitting node {node.node_id}: {e}")

        return node

    def _generate_rewrites(self, node: SegmentNode) -> SegmentNode:
        """Generate rewrite candidates for a node's chunk."""
        if not node.chunk_text:
            return node

        candidates = []

        for attempt in range(self.prompt_retries + 1):
            try:
                self.stats["llm_calls"] += 1
                current_candidates = []

                if self.rewrite_strategy == "multi_variant":
                    rewriter = dspy.Predict(MultiVariantRewriteChunk)
                    with dspy.settings.context(trace=[], lm=self.prompt_model, adapter=BoundedChatAdapter()):
                        pred = rewriter(text=node.chunk_text)

                    # Add concise variant if smaller
                    if pred.concise_summary and len(pred.concise_summary) < len(node.chunk_text):
                        current_candidates.append(
                            RewriteCandidate(
                                rewritten_text=pred.concise_summary,
                                target_compression_ratio=0.3,  # Estimate
                                generation_seed=None,
                            )
                        )

                    # Add detailed variant if smaller
                    if pred.detailed_summary and len(pred.detailed_summary) < len(node.chunk_text):
                        current_candidates.append(
                            RewriteCandidate(
                                rewritten_text=pred.detailed_summary,
                                target_compression_ratio=0.6,  # Estimate
                                generation_seed=None,
                            )
                        )

                    if not current_candidates:
                        logger.info(f"  Rewrite attempt {attempt + 1} failed: rewrites not smaller than original.")
                        continue

                else:
                    # Basic strategy using RewriteChunk
                    target_len = int(len(node.chunk_text) * self.target_compression_ratio)
                    rewriter = dspy.Predict(RewriteChunk)
                    with dspy.settings.context(trace=[], lm=self.prompt_model, adapter=BoundedChatAdapter()):
                        pred = rewriter(text=node.chunk_text, target_length=str(target_len))

                    if not pred.rewritten_text or len(pred.rewritten_text) >= len(node.chunk_text):
                        logger.info(f"  Rewrite attempt {attempt + 1} failed: rewrite not smaller than original.")
                        continue

                    current_candidates.append(
                        RewriteCandidate(
                            rewritten_text=pred.rewritten_text,
                            target_compression_ratio=self.target_compression_ratio,
                            generation_seed=None,
                        )
                    )

                candidates = current_candidates
                break

            except Exception as e:
                logger.warning(f"Rewrite generation failed for {node.node_id} (attempt {attempt + 1}): {e}")

        node.rewrite_candidates = candidates
        return node

    def _build_tree_parallel(self, text: str, depth: int, node_id: str) -> SegmentNode:
        """Build tree with parallel LLM calls for splits and rewrites."""
        root_node = SegmentNode(node_id=node_id, depth=depth, text=text)

        # Level-order traversal for splitting
        current_level_nodes = [root_node]

        while current_level_nodes:
            # Filter nodes that need splitting
            nodes_to_split = [
                n for n in current_level_nodes if n.depth < self.max_depth and len(n.text) >= self.min_chunk_chars
            ]

            if not nodes_to_split:
                break

            # Parallel split
            executor = ParallelExecutor(
                num_threads=self.tree_building_threads,
                disable_progress_bar=False,
                max_errors=100,  # Allow failures
            )

            logger.info(f"Parallel splitting {len(nodes_to_split)} nodes at depth {nodes_to_split[0].depth}...")

            executor.execute(self._generate_split, nodes_to_split)

            # Collect next level nodes
            next_level_nodes = []
            for node in nodes_to_split:
                if node.left_child:
                    next_level_nodes.append(node.left_child)
                if node.right_child:
                    next_level_nodes.append(node.right_child)

            current_level_nodes = next_level_nodes

        # After tree is built, generate rewrites for all chunks
        # Collect all nodes with chunks
        nodes_with_chunks = []
        stack = [root_node]
        while stack:
            node = stack.pop()
            if node.chunk_text:
                nodes_with_chunks.append(node)
            if node.left_child:
                stack.append(node.left_child)
            if node.right_child:
                stack.append(node.right_child)

        if nodes_with_chunks:
            logger.info(f"Parallel generating rewrites for {len(nodes_with_chunks)} chunks...")

            executor = ParallelExecutor(
                num_threads=self.tree_building_threads,
                disable_progress_bar=False,
                max_errors=100,
            )
            executor.execute(self._generate_rewrites, nodes_with_chunks)

        return root_node

    def _evaluate(self, program, eval_set):
        """Helper to evaluate a program."""
        with dspy.context(lm=self.task_model):
            evaluator = Evaluate(
                devset=eval_set,
                metric=self.metric,
                num_threads=self.num_threads,
                display_progress=False,
                display_table=False,
            )
            result = evaluator(program)

            # Handle EvaluationResult object if returned
            if hasattr(result, "score"):
                score = result.score
            else:
                score = result

            # Ensure it's in percentage form (0-100) if it's a ratio (0-1)
            if score <= 1.0:
                score = score * 100.0

            return score

    def _validate_reconstruction(self, original: str, reconstructed: str) -> bool:
        """
        Use LLM to validate that reconstructed text is semantically equivalent to original.
        """
        # Quick exact match check first (fast path)
        if original.strip() == reconstructed.strip():
            return True

        # Also check whitespace-normalized version before using LLM
        if original.replace(" ", "").replace("\n", "").replace("\t", "") == reconstructed.replace(" ", "").replace(
            "\n", ""
        ).replace("\t", ""):
            return True

        # LLM-based semantic validation
        validator = dspy.Predict(ValidateReconstruction)

        try:
            self.stats["llm_calls"] += 1
            with dspy.settings.context(trace=[], lm=self.prompt_model, adapter=BoundedChatAdapter()):
                result = validator(original_text=original, reconstructed_text=reconstructed)

            is_valid = getattr(result, "is_valid", "no").lower().strip()

            if is_valid == "no":
                reasoning = getattr(result, "reasoning", "")
                logger.info(f"  Reconstruction invalid: {reasoning}")

            return is_valid == "yes"

        except Exception as e:
            logger.warning(f"Error validating reconstruction: {e}")
            return False

    def _optimize_node(self, node: SegmentNode, root: SegmentNode, predictor, program, baseline_score, eval_set):
        """
        Apply optimization (cut/rewrite) to a single node.
        """
        # Only process if we have a chunk to cut/rewrite
        if not node.chunk_text:
            logger.info(f"Skipping processing for node {node.node_id} (no chunk text)")
            return

        logger.info(f"\nProcessing Node {node.node_id} (Depth {node.depth})")
        logger.info(f"  Chunk Text: {node.chunk_text!r}")

        # Attempt Cut
        if self._attempt_cut(node, root, predictor, program, baseline_score, eval_set):
            return

        # If cut failed, Attempt Rewrite
        self._attempt_rewrite(node, root, predictor, program, baseline_score, eval_set)

    def _process_tree_postorder(
        self, node: SegmentNode, root: SegmentNode, predictor, program, baseline_score, eval_set
    ):
        """
        Post-order traversal: Children -> Parent
        """
        self.stats["nodes_visited"] += 1

        # 1. Process children first
        if node.left_child:
            self._process_tree_postorder(node.left_child, root, predictor, program, baseline_score, eval_set)
        if node.right_child:
            self._process_tree_postorder(node.right_child, root, predictor, program, baseline_score, eval_set)

        # 2. Process this node
        self._optimize_node(node, root, predictor, program, baseline_score, eval_set)

    def _process_tree_preorder(
        self, node: SegmentNode, root: SegmentNode, predictor, program, baseline_score, eval_set
    ):
        """
        Pre-order traversal: Parent -> Children
        """
        self.stats["nodes_visited"] += 1

        # 1. Process this node first
        self._optimize_node(node, root, predictor, program, baseline_score, eval_set)

        # 2. Process children
        if node.left_child:
            self._process_tree_preorder(node.left_child, root, predictor, program, baseline_score, eval_set)
        if node.right_child:
            self._process_tree_preorder(node.right_child, root, predictor, program, baseline_score, eval_set)

    def _process_node(
        self,
        node: SegmentNode,
        root: SegmentNode,
        predictor,
        program,
        baseline_score,
        eval_set,
        strategy: str = "post_order",
    ):
        """
        Entry point for tree processing with specified strategy.
        """
        if strategy == "post_order":
            self._process_tree_postorder(node, root, predictor, program, baseline_score, eval_set)
        elif strategy == "pre_order":
            self._process_tree_preorder(node, root, predictor, program, baseline_score, eval_set)
        else:
            raise ValueError(f"Unknown traversal strategy: {strategy}")

    def _attempt_cut(self, node: SegmentNode, root: SegmentNode, predictor, program, baseline_score, eval_set) -> bool:
        if not self.enable_cutting:
            logger.info("  Skipping CUT (disabled)")
            return False

        logger.info("  Attempting CUT...")

        original_status = node.status
        node.status = "cut"

        # Apply change
        current_text = root.get_rendered_text()
        signature = get_signature(predictor)
        updated_signature = signature.with_instructions(current_text)
        set_signature(predictor, updated_signature)

        # Evaluate
        score = self._evaluate(program, eval_set)
        threshold = baseline_score + MODE_THRESHOLDS[self.quality_mode]

        if score >= threshold:
            logger.info(f"{GREEN}  Decision: CUT accepted at {node.node_id}{ENDC}")
            logger.info(
                f"  Score: {GREEN}{BOLD}{score:.1f}%{ENDC} (Baseline: {baseline_score:.1f}%, Threshold: {YELLOW}{threshold:.1f}%{ENDC})"
            )
            logger.info(f"  Saved: {BOLD}{len(node.chunk_text.split())}{ENDC} tokens (approx)")

            node.score_after = score
            node.saved_tokens = len(node.chunk_text.split())  # Approx
            self.stats["nodes_cut"] += 1
            return True
        logger.info(f"  Decision: CUT rejected at {node.node_id}")
        logger.info(f"  Score: {score:.1f}% (Threshold: {threshold:.1f}%)")

        # Revert
        node.status = original_status
        # Restore signature (important!)
        reverted_text = root.get_rendered_text()
        set_signature(predictor, signature.with_instructions(reverted_text))
        return False

    def _attempt_rewrite(
        self, node: SegmentNode, root: SegmentNode, predictor, program, baseline_score, eval_set
    ) -> bool:
        logger.info("  Attempting REWRITE...")

        original_status = node.status
        signature = get_signature(predictor)

        # Get candidates to try
        candidates_to_try = []
        if node.rewrite_candidates:
            candidates_to_try = [c.rewritten_text for c in node.rewrite_candidates]
        else:
            # Fallback to on-demand generation (single variant)
            rewriter = dspy.Predict(RewriteChunk)
            target_len = int(len(node.chunk_text) * self.target_compression_ratio)

            try:
                self.stats["llm_calls"] += 1
                with dspy.settings.context(trace=[], lm=self.prompt_model, adapter=BoundedChatAdapter()):
                    pred = rewriter(text=node.chunk_text, target_length=str(target_len))
                candidates_to_try.append(pred.rewritten_text)
            except Exception as e:
                logger.warning(f"Error rewriting at {node.node_id}: {e}")
                node.status = original_status
                return False

        # Try each candidate
        for i, rewritten in enumerate(candidates_to_try):
            logger.info(f"  Trying Rewrite Variant {i + 1}/{len(candidates_to_try)}: {rewritten!r}")

            # Apply change
            node.status = "rewritten"
            node.replacement_text = rewritten

            current_text = root.get_rendered_text()

            try:
                updated_signature = signature.with_instructions(current_text)
                set_signature(predictor, updated_signature)

                # Evaluate
                score = self._evaluate(program, eval_set)
                threshold = baseline_score + MODE_THRESHOLDS[self.quality_mode]

                if score >= threshold:
                    logger.info(f"{GREEN}  Decision: REWRITE accepted at {node.node_id} (Variant {i + 1}){ENDC}")
                    logger.info(
                        f"  Score: {GREEN}{BOLD}{score:.1f}%{ENDC} (Baseline: {baseline_score:.1f}%, Threshold: {YELLOW}{threshold:.1f}%{ENDC})"
                    )
                    logger.info(
                        f"  Saved: {BOLD}{len(node.chunk_text.split()) - len(rewritten.split())}{ENDC} tokens (approx)"
                    )

                    node.score_after = score
                    node.saved_tokens = len(node.chunk_text.split()) - len(rewritten.split())
                    self.stats["nodes_rewritten"] += 1
                    return True

                logger.info(f"  Decision: REWRITE rejected at {node.node_id} (Variant {i + 1})")
                logger.info(f"  Score: {score:.1f}% (Threshold: {threshold:.1f}%)")

            except Exception as e:
                logger.warning(f"Error evaluating rewrite at {node.node_id}: {e}")

        # If we get here, all variants failed
        # Revert
        node.status = original_status
        node.replacement_text = None
        reverted_text = root.get_rendered_text()
        set_signature(predictor, signature.with_instructions(reverted_text))
        return False
