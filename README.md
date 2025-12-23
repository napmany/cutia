<img width="1605" height="493" alt="cutia-3" src="https://github.com/user-attachments/assets/1951f7b6-2e05-4c5e-b2f3-17dd31123d02" />

<h1 align="center">CUTIA: Quality-Aware Prompt Compressor</h1>

<p align="center">
  <em>A prompt optimizer that cuts token usage while maintaining quality.</em>
</p>

[![PyPI - Version](https://img.shields.io/pypi/v/cutia)](https://pypi.org/project/cutia/) [![PyPI Downloads](https://static.pepy.tech/badge/cutia)](https://pepy.tech/projects/cutia) ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/napmany/cutia/publish.yml)

## Features

- **Tree-based Segmentation**: Recursively splits prompts into segments for fine-grained optimization
- **Cut-then-Rewrite Strategy**: Attempts to remove redundant content, then rewrites if cutting fails
- **Quality-Aware Compression**: Maintains quality thresholds during compression
- **Multi-Candidate Generation**: Generates multiple compression variants and chooses the best
- **DSPy Integration**: First-class support for DSPy programs via the DSPy adapter

## Installation

```bash
pip install cutia
```

## Usage

### DSPy Adapter

The DSPy adapter allows you to compress DSPy programs:

```python
import dspy
from cutia.adapters.dspy_adapter import CUTIA

# Configure models
# prompt_model generates rewrite candidates
prompt_model = dspy.LM(
    model="openai/gpt-4o-mini",
    max_tokens=10000,
    temperature=1,
)
# task_model runs the task/program for scoring and validation
task_model = dspy.LM(
    model="openai/gpt-4.1-nano",
    max_tokens=2000,
    temperature=1,
)

# Define your metric
def your_metric(example, prediction, trace=None):
    return example.output == prediction.output

# Create optimizer
optimizer = CUTIA(
    prompt_model=prompt_model,
    task_model=task_model,
    metric=your_metric,
)

# Compile your program
compressed_program = optimizer.compile(
    student=your_program,
    trainset=train_examples,
    valset=val_examples,
)
```

## Local AI

If you’re running CUTIA (or other prompt optimizers) against locally hosted LLMs, **vLLM** is a solid option for serving models: it supports **high-throughput** inference and handles **concurrent requests** efficiently.  
[**vLLM**](https://github.com/vllm-project/vllm)

If you’d like to use a separate **prompt model** from the **task model**, **llmsnap** can help by enabling **fast model switching** via vLLM’s sleep/wake mode—so you can swap models in seconds.  
[**llmsnap**](https://github.com/napmany/llmsnap)

## How It Works

1. **Tree Building**: The prompt is recursively split into segments (left, chunk, right)
2. **Node Processing**: For each node in the tree:
   - Attempt to **cut** the chunk entirely
   - If cutting fails quality check, attempt to **rewrite** the chunk
   - Keep original if both fail
3. **Multi-Candidate**: Generate multiple compression variants with different random seeds
4. **Selection**: Evaluate candidates on validation set and select the best

## Examples

### Strawberry Problem (Letter Counting)

Demonstrates prompt compression on a character counting task using the CharBench dataset.

See [src/cutia/examples/README.md](src/cutia/examples/README.md) for details.

## Development

### Development Installation

For development with testing and linting tools:

```bash
# Clone the repository
git clone https://github.com/napmany/cutia.git
cd cutia

# Install with development dependencies
uv sync --extra dev
```

### Running Tests

```bash
# Install development dependencies (if not already installed)
uv sync --extra dev

# Run tests
make test

```

### Code Quality

The project uses Ruff for linting and formatting, and Pyright for type checking:

```bash
# Run all checks (linting, formatting, and type checking)
make check
```

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

## Star History

> [!NOTE]
> ⭐️ Star this project to help others discover it!

[![Star History Chart](https://api.star-history.com/svg?repos=napmany/cutia&type=Date)](https://www.star-history.com/#napmany/cutia&Date)
