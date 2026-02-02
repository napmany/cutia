# CUTIA Data Models

> Freshness: 2026-01-31T17:45:00Z

## Core Data Structures

### SegmentNode (dspy_adapter.py:56)
Tree node for hierarchical prompt segmentation.

```python
@dataclass
class SegmentNode:
    node_id: str              # Unique identifier (e.g., "root.L.R")
    depth: int                # Tree depth level
    text: str                 # Full text at this node
    span_start: int = 0       # Character position start
    span_end: int = 0         # Character position end

    # Tree structure
    left_child: Optional["SegmentNode"] = None
    right_child: Optional["SegmentNode"] = None

    # Split info
    left_text: str | None = None      # Text before chunk
    chunk_text: str | None = None     # Candidate for removal/rewrite
    right_text: str | None = None     # Text after chunk
    chunk_reason: str | None = None   # Why chunk was selected

    # Decision info
    status: str = "pending"           # pending|cut|rewritten|kept
    replacement_text: str | None = None
    score_before: float = 0.0
    score_after: float = 0.0
    saved_tokens: int = 0

    # Rewrite options
    rewrite_candidates: list[RewriteCandidate] | None = None
    split_attempts: int = 0

    # Methods
    def get_rendered_text() -> str    # Reconstruct text based on status
    def _smart_delim(left, right) -> str  # Heuristic delimiter insertion
    def _merge_segments(left, right) -> str  # Handle double spaces
```

### RewriteCandidate (dspy_adapter.py:47)
Pre-generated rewrite option for a chunk.

```python
@dataclass
class RewriteCandidate:
    rewritten_text: str
    target_compression_ratio: float
    generation_seed: int | None = None
```

## Configuration Constants

### Quality Mode Thresholds (dspy_adapter.py:33)
```python
MODE_THRESHOLDS = {
    "strict": 0.0,       # No score degradation allowed
    "balanced": -5.0,    # Up to 5% drop allowed
    "aggressive": -10.0, # Up to 10% drop allowed
}
```

### Type Definitions
```python
QualityModeType = Literal["strict", "balanced", "aggressive"]
```

### CUTIA Constructor Parameters
```python
prompt_model=None            # LM for tree building/rewrites
task_model=None              # LM for task evaluation
metric=None                  # Quality metric function
max_depth=5                  # Max tree depth
min_chunk_chars=50           # Min chars to consider splitting
quality_mode="strict"        # Quality threshold mode
prompt_retries=2             # LLM retry attempts
target_compression_ratio=0.5 # Target for rewrites
num_candidates=4             # Parallel candidates
candidate_seed_offset=1000   # Seed spacing
seed=42                      # Base random seed
candidate_selection="weighted"  # Selection strategy
quality_weight=0.3           # Weight for quality score
compression_weight=0.7       # Weight for compression
node_decision_minibatch=True # Bootstrap aggregating
node_decision_minibatch_size=None  # Default: valset_size
top_k_candidates=2           # Top-K for trainset re-eval
traversal_strategy="pre_order"  # Tree traversal order
tree_building_threads=None   # Parallel threads
enable_cutting=True          # Allow chunk cutting
rewrite_strategy="basic"     # "basic"|"multi_variant"
```

## Output Schemas

### compression_stats (attached to program)
```python
{
    "original_total_tokens": int,
    "compressed_total_tokens": int,
    "original_total_chars": int,
    "compressed_total_chars": int,
    "compression_ratio": float,
    "token_compression_ratio": float,
    "mode": "CUTIA",
    "quality_mode": str,
    "per_signature_stats": [
        {
            "predictor_index": int,
            "original_chars": int,
            "compressed_chars": int,
            "compression_ratio": float,
            "original_tokens": int,
            "compressed_tokens": int,
            "token_compression_ratio": float,
            "nodes_visited": int,
            "nodes_cut": int,
            "nodes_rewritten": int,
            "baseline_score": float,
            "final_score": float,
        }
    ]
}
```

### compression_candidates (attached to program)
```python
[
    {
        "program": dspy.Module,
        "score": float,
        "seed": int,
        "compression_ratio": float,
        "original_tokens": int,
        "compressed_tokens": int,
        "node_decision_set_size": int,
        "final_eval_set_size": int,
        "candidate_baseline": float,
        "strategy": str,
        "trainset_score": float,  # After top-K re-eval
    }
]
```

## DSPy Signature Schemas

### ProposeChunk (Input/Output)
```
Input:
  instruction_to_analyze: str

Output:
  has_chunk: bool
  left: str (nullable)
  chunk: str (nullable)
  right: str (nullable)
  chunk_reason: str
```

### RewriteChunk (Input/Output)
```
Input:
  text: str
  target_length: str

Output:
  rewritten_text: str
```

### MultiVariantRewriteChunk (Input/Output)
```
Input:
  text: str

Output:
  concise_summary: str   # High compression
  detailed_summary: str  # Moderate compression
```

### ValidateReconstruction (Input/Output)
```
Input:
  original_text: str
  reconstructed_text: str

Output:
  reasoning: str
  is_valid: str ("yes"|"no")
```
