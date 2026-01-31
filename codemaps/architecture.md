# CUTIA Architecture Overview

> Freshness: 2026-01-31T17:45:00Z

## Project Type
Pure Python library (v0.0.2) for quality-aware prompt compression with DSPy integration.

## Directory Structure
```
cutia/
├── src/cutia/           # Main library source
│   ├── adapters/        # Framework adapters
│   │   └── dspy_adapter/   # DSPy teleprompter implementation
│   ├── utils/           # Shared utilities
│   └── examples/        # Usage examples
├── tests/               # Test suite
└── codemaps/            # Architecture documentation
```

## Core Components

### 1. CUTIA Optimizer (`adapters/dspy_adapter/dspy_adapter.py`)
- Tree-based prompt compression algorithm
- Implements DSPy `Teleprompter` interface
- Multi-candidate generation with quality thresholds
- Trainset minibatch exploration (bootstrap aggregating)
- Top-K candidate re-evaluation on trainset

### 2. BoundedChatAdapter (`adapters/dspy_adapter/bounded_chat_adapter.py`)
- Custom DSPy adapter for clear input/output boundaries
- Adds `[[ ## inputs_end ## ]]` marker after input fields
- Prevents LLM confusion between format instructions and data

### 3. Logging Utilities (`utils/logging_utils.py`)
- Configurable logging stream (`CUTIALoggingStream`)
- Enable/disable logging for library consumers
- Clean message-only formatting

## Dependencies
- **Required**: None (zero runtime dependencies)
- **Optional**: `dspy>=3.0.0` (for DSPy adapter)
- **Dev**: `pytest>=8.0.0`, `ruff>=0.3.0`, `pyright>=1.1.0`

## Data Flow
```
Input Program → Tree Building (parallel) → Node Processing (Cut/Rewrite) → Candidate Selection (top-K) → Compressed Program
```

## Integration Points
- DSPy Teleprompter interface
- LiteLLM-compatible language models (via DSPy)
- ParallelExecutor for concurrent tree building/rewrites
