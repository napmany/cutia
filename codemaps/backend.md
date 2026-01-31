# CUTIA Backend Structure

> Freshness: 2026-01-31T17:45:00Z

## Module Map

### src/cutia/
```
__init__.py                     # Package root, exports logging utilities
├── adapters/
│   ├── __init__.py             # Empty
│   └── dspy_adapter/
│       ├── __init__.py         # Exports: CUTIA
│       ├── dspy_adapter.py     # Main CUTIA class (1296 lines)
│       └── bounded_chat_adapter.py  # BoundedChatAdapter (140 lines)
├── utils/
│   ├── __init__.py             # Exports: configure_cutia_loggers, disable_logging, enable_logging
│   └── logging_utils.py        # CUTIALoggingStream, logging configuration (78 lines)
└── examples/
    ├── __init__.py             # Empty
    └── strawberry.py           # Letter counting example (731 lines)
```

## Key Classes

### CUTIA (dspy_adapter.py:227)
```python
class CUTIA(Teleprompter):
    """Tree-Structured Evaluate Cut-Then-Transform Compressor"""

    # Key params
    prompt_model: dspy.LM          # Model for tree building/rewrites
    task_model: dspy.LM            # Model for task evaluation
    metric: Callable               # Quality metric function
    quality_mode: str              # "strict"|"balanced"|"aggressive"
    num_candidates: int            # Parallel compression variants
    traversal_strategy: str        # "pre_order"|"post_order"|"random"
    node_decision_minibatch: bool  # Enable minibatch sampling for node decisions
    top_k_candidates: int          # Re-evaluate top-K on trainset (default: 2)
```

### SegmentNode (dspy_adapter.py:56)
```python
@dataclass
class SegmentNode:
    """Tree node for prompt segmentation"""
    node_id: str
    depth: int
    text: str
    left_child: Optional["SegmentNode"]
    right_child: Optional["SegmentNode"]
    status: str  # "pending"|"cut"|"rewritten"|"kept"
    rewrite_candidates: list[RewriteCandidate] | None  # Pre-generated options
```

### RewriteCandidate (dspy_adapter.py:47)
```python
@dataclass
class RewriteCandidate:
    """Pre-generated rewrite option for a chunk"""
    rewritten_text: str
    target_compression_ratio: float
    generation_seed: int | None
```

### DSPy Signatures (dspy_adapter.py)
- `ProposeChunk` (line 155): Split text into left/chunk/right
- `RewriteChunk` (line 189): Compress a chunk
- `MultiVariantRewriteChunk` (line 199): Generate concise/detailed variants
- `ValidateReconstruction` (line 212): Verify semantic equivalence

## Algorithm Flow

1. **compile()** → Entry point, manages candidates
2. **_create_node_decision_minibatch()** → Bootstrap aggregating sample
3. **_compress_with_seed()** → Per-candidate compression
4. **_build_tree()** → Parallel tree construction (level-order)
5. **_generate_split()** → LLM-based node splitting
6. **_generate_rewrites()** → Pre-compute rewrite candidates
7. **_process_node()** → Tree traversal (pre/post order)
8. **_optimize_node()** → Cut/rewrite decision per node
9. **_select_best_candidate()** → Top-K selection + trainset re-eval

## Candidate Selection Strategies
- `quality_first`: Best compression among threshold-passing candidates
- `weighted`: Combined score (quality_weight + compression_weight)
- `best_score`: Highest score regardless of compression

## External Dependencies

```python
# From dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.utils import get_signature, set_signature
from dspy.utils.parallelizer import ParallelExecutor
from dspy.adapters.chat_adapter import ChatAdapter
```

## Tests

```
tests/
├── conftest.py                  # Fixtures: clear_dspy_settings
└── adapters/dspy_adapter/
    └── test_cutia_basic.py      # Smoke tests: compile, quality_modes, traversal_strategies
```
