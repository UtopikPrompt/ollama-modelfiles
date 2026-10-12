# Examples

Collection of practical examples and code samples for the cookie cutter modelfile tool.

## Table of Contents

1. [CLI Examples](#cli-examples)
2. [Python API Examples](#python-api-examples)
3. [Interactive Mode Examples](#interactive-mode-examples)
4. [Batch Processing Examples](#batch-processing-examples)
5. [Custom Templates Examples](#custom-templates-examples)
6. [Advanced Use Cases](#advanced-use-cases)

---

## CLI Examples

### Basic Usage

```bash
# Create a Modelfile from Hugging Face model
cookie-cutter-modelfile create --model "huggingface-cli/clip-base-float16"

# Save to file
cookie-cutter-modelfile create --model "meta-llama/Meta-Llama-3-8B-Instruct" --output modelfile.txt

# Create chat model template
cookie-cutter-modelfile create \
  --model "mistralai/Mistral-7B-Instruct-v0.3" \
  --template chat \
  --output mistral-7b-instruct.modelfile
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
  --output llm-8b.modelfile

# Code generation parameters
cookie-cutter-modelfile create \
  --model "codellama/CodeLlama-34b-instruct" \
  --temperature 0.1 \
  --top-p 0.9 \
  --num_ctx 32768 \
  --output code-34b.modelfile
```

### Model Conversion

```bash
# Convert to Ollama model with default quantization
cookie-cutter-modelfile convert \
  --model "huggingface-cli/clip-base-float16" \
  --output clip-base:latest

# Convert with specific quantization
cookie-cutter-modelfile convert \
  --model "meta-llama/Meta-Llama-3-8B-Instruct" \
  --output llama3-8b \
  --quantize q4_k_m

# Convert with high precision
cookie-cutter-modelfile convert \
  --model "mistralai/Mistral-7B-Instruct-v0.3" \
  --output mistral-7b-instruct \
  --quantize q8_0
```

### Model Listing

```bash
# List all models
cookie-cutter-modelfile list

# List by category
cookie-cutter-modelfile list --category "text-generation"
cookie-cutter-modelfile list --category "text-classification"
cookie-cutter-modelfile list --category "question-answering"

# List with limit
cookie-cutter-modelfile list --limit 10
```

### Interactive Mode

```bash
# Run interactive wizard
cookie-cutter-modelfile interactive

# Sample interactive session:
$ cookie-cutter-modelfile interactive

Enter Hugging Face model ID (or "list" to browse): mistralai/Mistral-7B-Instruct-v0.3

Temperature (0.0-1.0, default 0.7): 0.7
Top-p (0.0-1.0, default 0.9): 0.9
Top-k (1-100, default 40): 40
Context length (256-32768, default 4096): 8192
Batch size (256-2048, default 512): 512
Repeat penalty (0.0-2.0, default 1.1): 1.1

Save to file? (y/n): y
File path (default: modelfile.txt): custom.modelfile

✓ Modelfile created!
```

---

## Python API Examples

### Basic Generation

```python
from cookiecutter_modelfile import generate

# Generate Modelfile from model ID
modelfile = generate("huggingface-cli/clip-base-float16")
print(modelfile)

# Save to file
with open("clip-base.modelfile", "w") as f:
    f.write(generate("huggingface-cli/clip-base-float16"))
```

### Custom Parameters

```python
from cookiecutter_modelfile import generate

# Generate with custom parameters
modelfile = generate(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    parameters={
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "num_ctx": 8192,
        "num_batch": 512,
        "repeat_penalty": 1.1
    }
)

# Custom template
modelfile = generate(
    "mistralai/Mistral-7B-Instruct-v0.3",
    template="chat",
    parameters={
        "temperature": 0.7,
        "top_p": 0.9
    }
)
```

### JSON Output

```python
from cookiecutter_modelfile import generate
import json

# Generate as JSON
modelfile_json = generate(
    "huggingface-cli/clip-base-float16",
    output_format="json"
)

# Parse and use
modelfile = json.loads(modelfile_json)
print(json.dumps(modelfile, indent=2))

# Save
with open("modelfile.json", "w") as f:
    f.write(modelfile_json)
```

### YAML Output

```python
from cookiecutter_modelfile import generate

# Generate as YAML
modelfile_yaml = generate(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    output_format="yaml"
)

# Parse and use
import yaml
modelfile = yaml.safe_load(modelfile_yaml)

# Save
with open("modelfile.yaml", "w") as f:
    f.write(modelfile_yaml)
```

### Model List

```python
from cookiecutter_modelfile import list_models

# List all models
models = list_models()
for model in models[:10]:
    print(f"{model.id}: {model.name}")

# List by category
text_models = list_models(category="text-generation")

# Paginated listing
models = list_models(limit=100, offset=0)
```

### Template System

```python
from cookiecutter_modelfile.templates import Template
from cookiecutter_modelfile.huggingface import fetch_model

# Get default template
default_template = Template.default()
print(default_template.to_string())

# Get chat template
chat_template = Template.chat()
print(chat_template.to_string())

# Render template
model = fetch_model("huggingface-cli/clip-base-float16")
rendered = chat_template.render({
    "model_id": model.id,
    "model_name": model.name,
    "temperature": 0.7,
    "top_p": 0.9
})
print(rendered)

# Load custom template
custom_template = Template.from_file("templates/custom.txt")
```

### Custom Modelfile

```python
from cookiecutter_modelfile.modelfile import Modelfile

# Create custom Modelfile
modelfile = Modelfile(name="my-custom-model")

# Add parameters
modelfile.add_parameter("temperature", 0.7)
modelfile.add_parameter("top_p", 0.9)
modelfile.add_parameter("top_k", 40)
modelfile.add_parameter("num_ctx", 8192)
modelfile.add_parameter("num_batch", 512)
modelfile.add_parameter("repeat_penalty", 1.1)
modelfile.add_parameter("repeat_last_n", 64)

# Convert to string
content = modelfile.to_string()
print(content)

# Save to file
with open("custom.modelfile", "w") as f:
    f.write(content)
```

### Fetch Model Info

```python
from cookiecutter_modelfile.huggingface import fetch_model, fetch_models

# Fetch single model
model = fetch_model("huggingface-cli/clip-base-float16")
print(f"Model: {model.name}")
print(f"Size: {model.size / (1024**3):.2f} GB")
print(f"License: {model.license}")

# Fetch multiple models
models = fetch_models(limit=5)
for model in models:
    print(f"{model.id}: {model.size} bytes")
```

---

## Interactive Mode Examples

### Complete Session

```bash
$ cookie-cutter-modelfile interactive

╔═══════════════════════════════════════════════════════════════════════════╗
║         Cookie Cutter Modelfile - Interactive Wizard                   ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Step 1/4: Select a Model                                              ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Enter Hugging Face model ID (or "list" to browse): mistralai/       ║
║  Mistral-7B-Instruct-v0.3                                              ║
║  ────────────────────────────────────────────────────────────────────   ║
║  Fetching model information...                                         ║
║  ✓ Model found: Mistral 7B Instruct v0.3                              ║
║  Size: 4.0 GB  |  License: MIT                                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════╗
║         Cookie Cutter Modelfile - Interactive Wizard                   ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Step 2/4: Customize Parameters                                       ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Temperature (0.0-1.0) - Controls randomness                          ║
║  Default: 0.7 | Your choice: 0.7                                      ║
║  Top-p (0.0-1.0) - Nucleus sampling                                   ║
║  Default: 0.9 | Your choice: 0.9                                      ║
║  Top-k (1-100) - Top-k sampling                                       ║
║  Default: 40 | Your choice: 40                                        ║
║  Context length (256-32768) - Maximum tokens                          ║
║  Default: 4096 | Your choice: 8192                                    ║
║  Batch size (256-2048) - Processing batch size                       ║
║  Default: 512 | Your choice: 512                                      ║
║  Repeat penalty (0.0-2.0) - Prevent repetition                       ║
║  Default: 1.1 | Your choice: 1.1                                      ║
╚═══════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════╗
║         Cookie Cutter Modelfile - Interactive Wizard                   ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Step 3/4: Save Modelfile                                             ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Save to file? (y/n) [y]: y                                           ║
║  File path (default: modelfile.txt): mistral-7b-instruct.modelfile    ║
╚═══════════════════════════════════════════════════════════════════════════╝

✓ Modelfile created successfully!

File: mistral-7b-instruct.modelfile

---

Modelfile Content:
MODELS_NAME: mistralai/Mistral-7B-Instruct-v0.3
PARAMETER temperature: 0.7
PARAMETER top_p: 0.9
PARAMETER top_k: 40
PARAMETER num_ctx: 8192
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

---

## Batch Processing Examples

### Convert Multiple Models

```python
import subprocess
import sys

# List of models to convert
models = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "huggingface-cli/clip-base-float16",
    "microsoft/Phi-3-mini-4k-instruct"
]

# Convert all models
for model_id in models:
    model_name = model_id.replace("/", "_").replace(".", "_")
    command = [
        sys.executable, "-m", "cookiecutter_modelfile",
        "convert",
        "--model", model_id,
        "--output", f"{model_name}:q4_k_m",
        "--quantize", "q4_k_m"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
```

### Batch Create Modelfiles

```python
import subprocess
import sys

# Models to process
models = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "codellama/CodeLlama-34b-instruct"
]

# Create modelfiles for all models
for model_id in models:
    filename = model_id.replace("/", "_").replace(".", "_") + ".modelfile"
    
    command = [
        sys.executable, "-m", "cookiecutter_modelfile",
        "create",
        "--model", model_id,
        "--output", filename
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ Created {filename}")
    else:
        print(f"✗ Failed to create {filename}")
        print(result.stderr, file=sys.stderr)
```

### Batch Convert to Different Quantizations

```python
import subprocess
import sys

# Convert to different quantizations
quantizations = ["q4_0", "q4_k_m", "q5_k_m", "q6_k", "q8_0"]
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

for quant in quantizations:
    model_name = model_id.replace("/", "_").replace(".", "_")
    command = [
        sys.executable, "-m", "cookiecutter_modelfile",
        "convert",
        "--model", model_id,
        "--output", f"{model_name}:{quant}",
        "--quantize", quant
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(f"{quant:8s}: {result.stdout.strip()}")
```

---

## Custom Templates Examples

### Chat Model Template

```bash
# Create custom chat template
mkdir -p templates

cat > templates/chat.txt << 'EOF'
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
  <|end_of_text|>
  <|eom|>
EOF

# Use custom template
cookie-cutter-modelfile create \
  --model "mistralai/Mistral-7B-Instruct-v0.3" \
  --template templates/chat.txt \
  --output mistral-7b-instruct.modelfile
```

### Code Model Template

```bash
cat > templates/code.txt << 'EOF'
MODELS_NAME: {{model_id}}
PARAMETER temperature: 0.1
PARAMETER top_p: 0.9
PARAMETER top_k: 50
PARAMETER num_ctx: 32768
PARAMETER num_batch: 512
PARAMETER num_thread: 4
PARAMETER repeat_penalty: 1.1
PARAMETER repeat_last_n: 64
PARAMETER mirostat: 0
PARAMETER mirostat_eta: 0.1
PARAMETER penalty_stop: 0.05
PARAMETER stop:
  <|user|>
  <|assistant|>
  <|ignore|>
EOF
```

### Image Model Template

```bash
cat > templates/image.txt << 'EOF'
MODELS_NAME: {{model_id}}
PARAMETER temperature: 0.5
PARAMETER num_ctx: 4096
PARAMETER num_batch: 512
PARAMETER num_thread: 4
PARAMETER repeat_penalty: 0.0
PARAMETER repeat_last_n: 0
PARAMETER mirostat: 0
PARAMETER mirostat_eta: 0.1
PARAMETER penalty_stop: 0.0
PARAMETER stop:
EOF
```

---

## Advanced Use Cases

### Model Comparison

```python
from cookiecutter_modelfile.huggingface import fetch_models

# Fetch and compare models
models = fetch_models(category="text-generation", limit=10)

# Display comparison
print(f"{'Model':<40} {'Size':<12} {'License':<15}")
print("-" * 70)

for model in models:
    size_str = f"{model.size / (1024**3):.2f} GB" if model.size else "N/A"
    print(f"{model.id:<40} {size_str:<12} {model.license or 'Unknown':<15}")
```

### Template Registry

```python
from cookiecutter_modelfile.templates import Template
from cookiecutter_modelfile.modelfile import Modelfile

# Create template registry
class TemplateRegistry:
    def __init__(self):
        self.templates = {}
    
    def register(self, name: str, template: Template):
        self.templates[name] = template
    
    def get(self, name: str) -> Template:
        return self.templates[name]

# Register custom templates
registry = TemplateRegistry()
registry.register("chat", Template.chat())
registry.register("code", Template.from_file("templates/code.txt"))
registry.register("image", Template.from_file("templates/image.txt"))

# Use registered template
modelfile = Modelfile(name="my-model")
chat_template = registry.get("chat")
modelfile.name = chat_template.render({
    "model_id": "huggingface-cli/clip-base-float16"
})
```

### Custom Parameter Validation

```python
from cookiecutter_modelfile.utils import validate_parameters

# Validate parameters before generation
def create_modelfile(model_id: str, parameters: dict):
    # Validate parameters
    try:
        validated = validate_parameters(parameters)
    except ValidationError as e:
        print(f"Invalid parameters: {e}")
        return None
    
    # Generate modelfile
    return generate(model_id, parameters=validated)

# Safe usage
result = create_modelfile(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    {
        "temperature": 0.7,      # Valid
        "top_p": 0.9,            # Valid
        "invalid_param": "value" # Will be caught
    }
)
```

---

## Quick Reference Card

```bash
# CREATE
cookie-cutter-modelfile create --model "hf/model" --output modelfile.txt

# CONVERT
cookie-cutter-modelfile convert --model "hf/model" --output model:q4_k_m

# LIST
cookie-cutter-modelfile list --category "text-generation"

# INTERACTIVE
cookie-cutter-modelfile interactive

# PYTHON
python -m cookiecutter_modelfile create --model "hf/model" --output modelfile.txt
```

---

*Examples last updated: 2026-10-12*
