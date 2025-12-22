<img width="1605" height="493" alt="cutia-3" src="https://github.com/user-attachments/assets/1951f7b6-2e05-4c5e-b2f3-17dd31123d02" />

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
git clone https://github.com/napmany/cutia.git
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
    num_candidates=4,
)

# Compile your program
compressed_program = optimizer.compile(
    student=your_program,
    trainset=train_examples,
    valset=val_examples,
)
```

## Examples

### Strawberry Problem (Letter Counting)

Demonstrates prompt compression on a character counting task using the CharBench dataset.

See [src/cutia/examples/README.md](src/cutia/examples/README.md) for details.

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
