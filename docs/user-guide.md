# User Guide

Complete user guide for the cookie cutter modelfile creation tool.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Creating Modelfiles](#creating-modelfiles)
5. [Command Reference](#command-reference)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Introduction

This tool simplifies the process of creating Ollama Modelfiles from Hugging Face models. It handles model discovery, parameter configuration, and file generation automatically.

### What is a Modelfile?

A Modelfile is a simple text file that defines a custom Ollama model. It includes:
- **Model name**: Identifier for the model
- **Parameters**: Configuration options like temperature, context length, etc.

### Why Use This Tool?

- ✅ Automatic model discovery from Hugging Face
- ✅ Pre-configured templates for different use cases
- ✅ One-command model conversion
- ✅ Interactive mode for beginners
- ✅ Custom templates for special requirements

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install the Tool

```bash
pip install .
```

Or if you're using a virtual environment:

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Add to PATH (Windows)
set PATH=%PATH%;venv\Scripts
```

### Verify Installation

```bash
cookie-cutter-modelfile --version
```

---

## Quick Start

### Create a Modelfile from Hugging Face Model

```bash
# Basic usage
cookie-cutter-modelfile create --model "huggingface-cli/clip-base-float16"

# Save to file
cookie-cutter-modelfile create --model "meta-llama/Meta-Llama-3-8B-Instruct" --output modelfile.txt
```

### Convert to Ollama Model

```bash
# Convert with default quantization
cookie-cutter-modelfile convert --model "huggingface-cli/clip-base-float16" --output clip-base:latest

# Convert with specific quantization
cookie-cutter-modelfile convert --model "huggingface-cli/clip-base-float16" --output clip-base --quantize q4_k_m
```

### Interactive Mode

```bash
# Run the interactive wizard
cookie-cutter-modelfile interactive
```

---

## Creating Modelfiles

### Template Types

The tool provides three built-in templates:

| Template | Use Case | Description |
|----------|----------|-------------|
| `chat` | Chat models | Llama, Mistral, Gemma, etc. |
| `completion` | Text generation | General text completion |
| `image` | Image models | Stable Diffusion, Flux, etc. |

### Using Custom Templates

```bash
# Load custom template
cookie-cutter-modelfile create \
  --model "your/model" \
  --template "templates/custom.txt" \
  --output modelfile.txt
```

### Custom Parameters

```bash
# Override default parameters
cookie-cutter-modelfile create \
  --model "meta-llama/Meta-Llama-3-8B-Instruct" \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 40 \
  --num_ctx 8192 \
  --output modelfile.txt
```

---

## Command Reference

### `create` - Create a Modelfile

```bash
cookie-cutter-modelfile create

Required Arguments:
  --model, -m <MODEL>    Hugging Face model identifier (required)

Optional Arguments:
  --output, -o <FILE>    Output file path (default: stdout)
  --template, -t <FILE>  Custom template file path
  --temperature <FLOAT>  Default temperature (default: 0.7)
  --top-p <FLOAT>        Default top_p (default: 0.9)
  --top-k <INT>          Default top_k (default: 40)

Examples:
  # Create basic modelfile
  cookie-cutter-modelfile create --model "huggingface-cli/clip-base-float16"
  
  # Create with custom parameters
  cookie-cutter-modelfile create \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --temperature 0.7 \
    --output modelfile.txt
```

### `convert` - Convert to Ollama Model

```bash
cookie-cutter-modelfile convert

Required Arguments:
  --model, -m <MODEL>    Hugging Face model identifier (required)
  --output, -o <NAME>    Output model name (required)

Optional Arguments:
  --quantize <LEVEL>     Quantization level (default: q4_k_m)

Examples:
  # Convert to Ollama model
  cookie-cutter-modelfile convert \
    --model "huggingface-cli/clip-base-float16" \
    --output clip-base:latest
  
  # Convert with quantization
  cookie-cutter-modelfile convert \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --output llama3-8b \
    --quantize q4_k_m
```

### `list` - List Available Models

```bash
cookie-cutter-modelfile list

Optional Arguments:
  --category, -c <CAT>  Filter by category (e.g., text-generation)

Examples:
  # List all models
  cookie-cutter-modelfile list
  
  # List text generation models
  cookie-cutter-modelfile list --category "text-generation"
```

### `interactive` - Interactive Wizard

```bash
cookie-cutter-modelfile interactive
```

---

## Advanced Features

### Model Comparison

Compare multiple models side-by-side:

```bash
cookie-cutter-modelfile compare "model1" "model2" "model3"
```

### Batch Processing

Process multiple models:

```bash
# Convert multiple models
models=("model1" "model2" "model3")
for model in "${models[@]}"; do
  cookie-cutter-modelfile convert \
    --model "$model" \
    --output "${model%/*}_q4_k_m:latest"
done
```

### Custom Templates

Create a custom template:

```bash
# templates/custom.txt
MODELS_NAME: {{model_id}}
PARAMETER temperature: {{temperature}}
PARAMETER top_p: {{top_p}}
PARAMETER top_k: {{top_k}}
PARAMETER num_ctx: {{num_ctx}}
PARAMETER num_batch: 512
PARAMETER num_thread: 4
PARAMETER repeat_penalty: 1.1
PARAMETER repeat_last_n: 64
PARAMETER mirostat: 0
PARAMETER mirostat_eta: 0.1
PARAMETER penalty_stop: 0.05
PARAMETER stop:
  <s>
  </s>
  <|eot_id|>
```

### JSON/YAML Export

Export to structured formats:

```python
from cookiecutter_modelfile import generate

modelfile = generate(
    model_id="huggingface-cli/clip-base-float16",
    output_format="json"
)

# Or YAML
modelfile = generate(
    model_id="huggingface-cli/clip-base-float16",
    output_format="yaml"
)
```

---

## Troubleshooting

### Common Issues

#### Issue: "Module not found"

```bash
# Install dependencies
pip install -r requirements.txt
```

#### Issue: "Network error"

```bash
# Check internet connection
ping huggingface.co

# Or use offline mode (if supported)
```

#### Issue: "Model not found"

```bash
# Verify model exists
cookie-cutter-modelfile list --category "text-generation"

# Check model ID
cookie-cutter-modelfile create --model "your/model" --output debug.txt
cat debug.txt
```

#### Issue: "Permission denied"

```bash
# Run with elevated privileges (Windows)
cookie-cutter-modelfile --no-sandbox create --model "..."

# Or adjust file permissions
chmod 644 modelfile.txt
```

### Debug Mode

```bash
# Enable verbose output
cookie-cutter-modelfile --debug create --model "..."
```

---

## Best Practices

### Parameter Recommendations

| Parameter | Range | Typical Value | Use Case |
|-----------|-------|---------------|----------|
| `temperature` | 0.0 - 1.0 | 0.7 | Creative writing |
| `top_p` | 0.0 - 1.0 | 0.9 | Balanced generation |
| `top_k` | 1 - 100 | 40 | Diverse tokens |
| `num_ctx` | 256 - 32768 | 4096 | Context length |
| `repeat_penalty` | 0.0 - 2.0 | 1.1 | Prevent repetition |

### Model Selection Tips

1. **For Chat**: Use `chat` template with `temperature: 0.7`
2. **For Code**: Use `completion` template with `temperature: 0.1`
3. **For Creative Writing**: Use `chat` template with `temperature: 0.8-0.9`
4. **For Summarization**: Use `completion` template with `temperature: 0.3`

### Performance Tips

```bash
# Optimize for memory
--num_ctx 2048
--num_thread 4

# Optimize for speed
--num_batch 512

# Optimize for quality
--num_ctx 8192
--temperature 0.7
```

---

## Getting Help

- **GitHub Issues**: [Open an issue](https://github.com/UtopikPrompt/ollama-modelfiles/issues)
- **Documentation**: [Full docs](https://github.com/UtopikPrompt/ollama-modelfiles/tree/main/docs)
- **Examples**: [Code samples](./examples.md)

---

*Last updated: 2026-10-12*
