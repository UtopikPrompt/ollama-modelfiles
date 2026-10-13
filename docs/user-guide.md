# User Guide

Complete user guide for the cookie cutter modelfile creation tool.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Creating Modelfiles](#creating-modelfiles)
5. [Command Reference](#command-reference)
6. [Configuration](#configuration)
7. [Authentication](#authentication)
8. [Advanced Features](#advanced-features)
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
  --hardware <NAME>      Hardware profile name (see [Configuration](#configuration))

Examples:
  # Create basic modelfile
  cookie-cutter-modelfile create --model "huggingface-cli/clip-base-float16"
  
  # Create with custom parameters
  cookie-cutter-modelfile create \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --temperature 0.7 \
    --output modelfile.txt
  
  # Create with a custom hardware profile
  cookie-cutter-modelfile create \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --hardware cpu-only
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

## Authentication

The Hugging Face client (`HFClient`) authenticates against the Hugging Face Hub
using a `HF_TOKEN`. This token is resolved automatically, in the following order
of precedence (the first non-empty value wins):

1. **Explicit token** — an `HF_TOKEN` value passed programmatically to the
   client (or a flag accepted by the CLI).
2. **Environment variable** — the `HF_TOKEN` environment variable.
3. **`.env` file** — an `HF_TOKEN` entry in a `.env` file located in the current
   working directory.

If none of these sources provide a token, the client is created without a token
and lets `huggingface_hub` handle the "no token" behavior (public models only).

### Create a `.env` file

Storing your token in a `.env` file is convenient because it is picked up
automatically whenever the tool is run from that directory, without needing to
export an environment variable or run `huggingface-cli login`.

```bash
# Create a .env file in your project directory
HF_TOKEN="your_huggingface_token"
```

You can create the file with your editor or any `.env` helper:

```bash
# Using python-dotenv (if installed)
python -c "from dotenv import load_dotenv; load_dotenv('.env')"

# Or with a simple here-doc
cat > .env <<'EOF'
HF_TOKEN="your_huggingface_token"
EOF
```

### Using an environment variable

Alternatively, export the token in your shell session:

```bash
# Windows (PowerShell)
$env:HF_TOKEN="your_huggingface_token"

# macOS / Linux
export HF_TOKEN="your_huggingface_token"
```

### Where to get a token

Create a token on the [Hugging Face settings page](https://huggingface.co/settings/tokens),
then copy the access token. Use a token with at least read/write permissions for
the models you intend to access.

---

## Configuration

This tool reads a configuration file (default: `config.yml`) that lets
you set the **Ollama server URL** and the **hardware profiles** used to
automatically calculate `num_thread` / `num_gpu_layers` for the generated
Modelfile.

### The configuration file

```yaml
# config.yml
ollama:
  url: http://localhost:11434   # Ollama server URL

hardware:
  default: workstation-gpu       # hardware profile applied by default
  profiles:
    workstation-gpu:
      devices: ["GPU:0"]
      num_threads: 0             # 0 = auto-detect from CPU count
      gpu_layers: -1             # -1 = auto-detect layers
    cpu-only:
      devices: ["CPU"]
      num_threads: 8
      gpu_layers: 0
```

#### Ollama URL

Set the Ollama server URL so the tool invokes `ollama --host <url>` when the
value differs from the default `http://localhost:11434`.

You can set it in three ways (precedence from highest to lowest):

1. **Configuration file** — the `ollama.url` key (or the `OLLAMA_URL` /
   `OLLAMA_HOST` environment variable).
2. **Environment variable** — `OLLAMA_URL` or `OLLAMA_HOST`.
3. **Default** — `http://localhost:11434`.

### Ollama URL via `.env`

You can store the Ollama URL in a `.env` file, in the same place as your
`HF_TOKEN`. The `.env` file is loaded automatically whenever the tool runs from
that directory, so you don't need to export an environment variable:

```bash
# .env
HF_TOKEN="your_huggingface_token"
OLLAMA_URL="http://localhost:11434"
# OLLAMA_HOST="ollama.example.com"   # alternative to OLLAMA_URL
```

> **Note:** An environment variable (`OLLAMA_URL` / `OLLAMA_HOST`) always wins
> over the same key in the `.env` file, because `python-dotenv` never overrides an
> existing environment variable.

### Hardware profiles

A **hardware profile** describes the machine that runs the model. It is applied
automatically whenever the tool generates a Modelfile, and it sets two
`PARAMETER` lines:

| Parameter | Purpose |
|-----------|---------|
| `num_thread` | Number of CPU threads used for inference. |
| `num_gpu_layers` | Number of model layers offloaded to the GPU. `-1` = auto-detect. |

Use the `--hardware` flag to override the default profile for a single command:

```bash
# Use the CPU-only profile for this command
cookie-cutter-modelfile create \
  --model "huggingface-cli/clip-base-float16" \
  --hardware cpu-only
```

The default profile can also be changed in the configuration file
(`hardware.default`), or overridden per-command with `--hardware`.

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
