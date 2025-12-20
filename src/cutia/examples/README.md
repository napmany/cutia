# CUTIA Strawberry Letter Counting Example

This example demonstrates how to use CUTIA to compress prompts for the "strawberry problem" - a character counting task that challenges many LLMs.

## Overview

The "strawberry problem" (counting how many times "r" appears in "strawberry") highlights tokenization limitations in LLMs. We use the [CharBench](https://huggingface.co/datasets/omriuz/CharBench) dataset to evaluate how well CUTIA can compress prompts while maintaining reasoning capabilities for this task.

## Prerequisites

### 1. Install Dependencies

This example requires the `datasets` library to load CharBench. Install it with the `full` extra:

```bash
uv sync --extra full
```

### 2. Environment Setup

The example relies on environment variables for API access. A template is provided in `.env.example`.

1.  Create a `.env` file in the same directory as the script (`src/cutia/examples/`):
    ```bash
    cp src/cutia/examples/.env.example src/cutia/examples/.env
    ```

2.  Edit `src/cutia/examples/.env` with your provider details:

    **For LocalAI (Default):**
    ```ini
    LOCALAI_BASE_URL=http://localhost:8080/v1
    LOCALAI_API_KEY=probably-not-needed
    ```

    **For OpenAI:**
    ```ini
    OPENAI_API_KEY=sk-...
    ```

    **For OpenRouter:**
    ```ini
    OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
    OPENROUTER_API_KEY=sk-...
    ```

## Running the Example

Run the example script using `uv`:

```bash
uv run python -m cutia.examples.strawberry
```

### Command Line Arguments

You can customize the execution with the following arguments:

- `--ai-provider`: Choose the AI provider.
    - `localai` (default): Runs `gpt-oss-20b` for prompt optimization and `Qwen3-0.6B` for task inference via your local OpenAI-compatible server.
    - `openai`: Uses OpenAI's `gpt-4o-mini` for prompt optimization and `gpt-5-nano` for task inference.
    - `openrouter`: Uses OpenRouter's `gpt-4o-mini` for prompt optimization and `gpt-5-nano` for task inference.

**Example: Using LocalAI**
```bash
uv run python -m cutia.examples.strawberry --ai-provider localai
```

**Example: Using OpenAI**
```bash
uv run python -m cutia.examples.strawberry --ai-provider openai
```

**Example: Using OpenRouter**
```bash
uv run python -m cutia.examples.strawberry --ai-provider openrouter
```

## What It Does

1. **Loads Data**: Downloads the CharBench dataset and filters for the character frequency counting task.
2. **Baseline**: Runs an uncompressed program to establish baseline accuracy.
3. **Compression**: Uses CUTIA to compress the prompt, removing redundant instructions and examples while preserving the core reasoning logic.
4. **Evaluation**: Compares the accuracy of the compressed program against the baseline.
