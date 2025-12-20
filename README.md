<img width="1728" height="624" alt="cutia-logo" src="https://github.com/user-attachments/assets/3c114b55-44be-4299-9a53-d554bb734e0d" />

# CUTIA - Cut-Then-Implement-Augment Prompt Compressor

CUTIA is a tree-based prompt compression library that uses a cut-then-transform strategy to compress prompts while maintaining quality.

## Features

- **Tree-based Segmentation**: Recursively splits prompts into segments for fine-grained optimization
- **Cut-then-Rewrite Strategy**: Attempts to remove redundant content, then rewrites if cutting fails
- **Quality-Aware Compression**: Maintains quality thresholds during compression
- **Multi-Candidate Generation**: Generates multiple compression variants with different random seeds
- **DSPy Integration**: First-class support for DSPy programs via the DSPy adapter

## Installation

### Basic Installation

```bash
pip install cutia
```

### Development Installation

For development with testing and linting tools:

```bash
# Clone the repository
git clone <repository-url>
cd cutia

# Install with development dependencies
uv sync --extra dev
```

## Usage

### DSPy Adapter

The DSPy adapter allows you to compress DSPy programs:

```python
import dspy
from cutia.adapters.dspy_adapter import CUTIA

# Configure models
prompt_model = dspy.LM("gpt-4o-mini")
task_model = dspy.LM("gpt-4o-mini")

# Define your metric
def your_metric(example, prediction, trace=None):
    return example.output == prediction.output

# Create optimizer
optimizer = CUTIA(
    prompt_model=prompt_model,
    task_model=task_model,
    metric=your_metric,
    quality_mode="strict",  # "strict", "balanced", or "aggressive"
    target_compression_ratio=0.5,
    num_candidates=4,
    traversal_strategy="pre_order",  # "pre_order", "post_order", or "random"
)

# Compile your program
compressed_program = optimizer.compile(
    student=your_program,
    trainset=train_examples,
    valset=val_examples,
)
```

### Quality Modes

CUTIA supports three quality modes:

- **`"strict"`**: No score degradation allowed (threshold: baseline + 0.0%)
  - Use case: Safety-critical prompts, zero quality loss tolerance
  - Expected compression: 10-20%

- **`"balanced"`**: Moderate degradation allowed (threshold: baseline - 5.0%) - **Default**
  - Use case: Most applications, good quality/compression balance
  - Expected compression: 25-40%

- **`"aggressive"`**: Larger degradation allowed (threshold: baseline - 10.0%)
  - Use case: Maximum compression priority, quality less critical
  - Expected compression: 40-60%

### Traversal Strategies

- **`"post_order"`**: Process children before parent (bottom-up)
- **`"pre_order"`**: Process parent before children (top-down)
- **`"random"`**: Randomly choose between post-order and pre-order for each candidate

## Development

### Running Tests

The project uses pytest for testing. All tests are designed to run without making actual LLM calls.

```bash
# Install development dependencies (if not already installed)
uv sync --extra dev

# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/adapters/dspy_adapter/test_cutia_basic.py

# Run specific test function
uv run pytest tests/adapters/dspy_adapter/test_cutia_basic.py::test_cutia_basic_compile

# Run tests with coverage (if pytest-cov installed)
uv run pytest tests/ --cov=cutia --cov-report=term-missing
```

### Code Quality

The project uses Ruff for linting and formatting, and Pyright for type checking:

```bash
# Linting
ruff check src/

# Formatting
ruff format src/

# Type checking
uv run pyright

# Run all checks (linting, formatting, and type checking)
make check
```

Alternatively, use the Makefile commands:

```bash
# Individual checks
make lint          # Run linting only
make format        # Run formatting only
make typecheck     # Run type checking only

# Combined checks
make check         # Run all quality checks (linting, formatting, type checking)
make fix           # Auto-fix linting and formatting issues
```

### Type Checking

The project uses Pyright for static type checking. The configuration is in `pyproject.toml` under `[tool.pyright]`.

```bash
# Run type checking manually
uv run pyright

# Run in watch mode (useful during development)
make typecheck-watch
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## How It Works

1. **Tree Building**: The prompt is recursively split into segments (left, chunk, right)
2. **Node Processing**: For each node in the tree:
   - Attempt to **cut** the chunk entirely
   - If cutting fails quality check, attempt to **rewrite** the chunk
   - Keep original if both fail
3. **Multi-Candidate**: Generate multiple compression variants with different random seeds
4. **Selection**: Evaluate candidates on validation set and select the best

## Dependencies

### Core
- No required dependencies for the base library

### Optional: DSPy Adapter
- `dspy-ai>=3.0.0` - For DSPy integration

### Development
- `pytest>=8.0.0` - Testing framework
- `ruff>=0.3.0` - Linting and formatting
- `pyright>=1.1.0` - Static type checking
- `pre-commit` - Git hooks

Install optional dependencies:

```bash
# For testing
uv sync --extra test

# For development (includes test dependencies)
uv sync --extra dev
```

## Future Plans

- Framework-agnostic core implementation (not tied to DSPy)
- Additional adapters for other frameworks and platforms (LangChain, MLflow, etc.)
- Standalone Python API for direct use
- Enhanced chunking strategies

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `uv run pytest tests/`
2. Code is formatted: `ruff format src/`
3. No linting errors: `ruff check src/`
4. Type checking passes: `uv run pyright` (optional but recommended)
5. Pre-commit hooks pass: `pre-commit run --all-files`
6. Add tests for new features

Or run `make check` to verify linting, formatting, and type checking in one command.

**Note:** Type checking is verified by `make check` but not enforced in pre-commit hooks, allowing for faster commits while still maintaining quality standards.
