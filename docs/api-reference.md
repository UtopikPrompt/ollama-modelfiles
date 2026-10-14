# API Reference

Complete API documentation for the cookie cutter modelfile tool.

## Table of Contents

1. [CLI API](#cli-api)
2. [Python API](#python-api)
3. [Template API](#template-api)
4. [Hugging Face API](#hugging-face-api)
5. [Data Models](#data-models)
6. [Configuration Models](#configuration-models)
7. [Context Estimation](#context-estimation)

---

## CLI API

### Entry Point

```bash
cookie-cutter-modelfile <command> [options]
```

### Commands

#### `create`

Create a Modelfile from a Hugging Face model.

```bash
cookie-cutter-modelfile create

Arguments:
  model (string)    Hugging Face model identifier
                    Format: "owner/repo" or "repo/suffix"

Options:
  -m, --model       Model identifier (required)
  -o, --output      Output file path (default: stdout)
  -t, --template    Custom template file path
  --temperature     Default temperature (default: 0.7)
  --top-p           Default top_p (default: 0.9)
  --top-k           Default top_k (default: 40)
  --help            Show this help message
```

#### `convert`

Convert a Hugging Face model to an Ollama model.

```bash
cookie-cutter-modelfile convert

Arguments:
  model (string)    Hugging Face model identifier
  output (string)   Output model name
                    Format: "model_name:tag"

Options:
  -m, --model       Model identifier (required)
  -o, --output      Output model name (required)
  --quantize        Quantization level
                    Options: q4_0, q4_1, q4_k_m, q5_k_m, q6_k, q8_0
                    (default: q4_k_m)
  --help            Show this help message
```

#### `list`

List available models from Hugging Face.

```bash
cookie-cutter-modelfile list

Arguments:
  None

Options:
  -c, --category    Filter by category
  --help            Show this help message
```

#### `interactive`

Run the interactive wizard.

```bash
cookie-cutter-modelfile interactive

Arguments:
  None

Options:
  --help            Show this help message
```

---

## Python API

### Main Module

```python
from cookiecutter_modelfile import generate, convert, list_models
```

### `generate()`

Generate a Modelfile from a model ID.

```python
def generate(
    model_id: str,
    template: str = "default",
    parameters: dict = None,
    output_format: str = "text"
) -> str | dict | list:
    """
    Generate a Modelfile from a Hugging Face model ID.
    
    Args:
        model_id: Hugging Face model identifier
        template: Template to use (default, chat, completion, image)
        parameters: Custom parameters dict
        output_format: Output format (text, json, yaml)
    
    Returns:
        Generated Modelfile content
    
    Examples:
        # Basic generation
        modelfile = generate("huggingface-cli/clip-base-float16")
        
        # With custom parameters
        modelfile = generate(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            parameters={
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 8192
            }
        )
        
        # JSON output
        json_modelfile = generate(
            "huggingface-cli/clip-base-float16",
            output_format="json"
        )
        
        # YAML output
        yaml_modelfile = generate(
            "huggingface-cli/clip-base-float16",
            output_format="yaml"
        )
    """
```

### `convert()`

Convert a Modelfile to an Ollama model.

```python
def convert(
    model_id: str,
    output_name: str,
    quantize: str = "q4_k_m",
    skip_verify: bool = False
) -> str:
    """
    Convert a Hugging Face model to an Ollama model.
    
    Args:
        model_id: Hugging Face model identifier
        output_name: Output model name
        quantize: Quantization level
        skip_verify: Skip verification after conversion
    
    Returns:
        Status message on success
    
    Examples:
        # Basic conversion
        status = convert("huggingface-cli/clip-base-float16", "clip-base:latest")
        
        # With specific quantization
        status = convert(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "llama3-8b",
            quantize="q4_k_m"
        )
    """
```

### `list_models()`

List available models from Hugging Face.

```python
def list_models(
    category: str = None,
    limit: int = 50,
    offset: int = 0
) -> list[dict]:
    """
    List models from Hugging Face.
    
    Args:
        category: Filter by category
        limit: Maximum number of results
        offset: Pagination offset
    
    Returns:
        List of model information dictionaries
    
    Examples:
        # List all models
        models = list_models()
        
        # List text generation models
        models = list_models(category="text-generation")
        
        # Paginated listing
        models = list_models(limit=100, offset=0)
    """
```

---

## Template API

### `Template` Class

```python
from cookiecutter_modelfile.templates import Template
```

### Available Templates

```python
# Default template
Template.default()

# Chat template
Template.chat()

# Completion template
Template.completion()

# Image template
Template.image()

# Custom template
Template.from_file("path/to/template.txt")
```

### Template Methods

```python
class Template:
    """Template class for Modelfile generation."""
    
    @classmethod
    def default(cls) -> Template
        """Return default template."""
    
    @classmethod
    def chat(cls) -> Template
        """Return chat template."""
    
    @classmethod
    def completion(cls) -> Template
        """Return completion template."""
    
    @classmethod
    def image(cls) -> Template
        """Return image template."""
    
    @classmethod
    def from_file(cls, path: str) -> Template
        """Load template from file."""
    
    def render(self, data: dict) -> str
        """Render template with data."""
    
    def to_string(self) -> str
        """Convert template to string."""
    
    def to_dict(self) -> dict
        """Convert template to dictionary."""
```

---

## Hugging Face API

### `ModelInfo` Data Model

```python
from cookiecutter_modelfile.huggingface import ModelInfo
```

```python
class ModelInfo:
    """Model information from Hugging Face."""
    
    id: str
    name: str
    description: str
    tags: list[str]
    license: str | None
    authors: list[str]
    size: int | None
    sha: str | None
    created_at: str | None
    
    @property
    def hf_id(self) -> str
        """Hugging Face model ID."""
    
    @property
    def ollama_id(self) -> str
        """Ollama-compatible model ID."""
```

### `fetch_model()`

```python
def fetch_model(model_id: str) -> ModelInfo:
    """
    Fetch model information from Hugging Face.
    
    Args:
        model_id: Hugging Face model identifier
    
    Returns:
        ModelInfo object
    
    Raises:
        ModelNotFoundError: Model not found
        NetworkError: Network-related errors
    """
```

### `fetch_models()`

```python
def fetch_models(
    category: str = None,
    limit: int = 50,
    offset: int = 0
) -> list[ModelInfo]:
    """
    Fetch list of models from Hugging Face.
    
    Args:
        category: Filter by category
        limit: Maximum number of results
        offset: Pagination offset
    
    Returns:
        List of ModelInfo objects
    """
```

---

## Data Models

### `ModelfileInfo`

```python
class ModelfileInfo:
    """Information about a generated Modelfile."""
    
    name: str
    model_id: str
    parameters: dict[str, Any]
    template: str
    created_at: datetime
    
    def to_dict(self) -> dict
        """Convert to dictionary."""
    
    def to_json(self) -> str
        """Serialize to JSON."""
```

### `Modelfile`

```python
class Modelfile:
    """Represents a Modelfile."""
    
    name: str
    parameters: dict[str, Any]
    
    def add_parameter(self, name: str, value: Any) -> None
        """Add a parameter."""
    
    def remove_parameter(self, name: str) -> None
        """Remove a parameter."""
    
    def to_string(self) -> str
        """Convert to string."""
    
    def to_dict(self) -> dict
        """Convert to dictionary."""
```

---

## Error Handling

### Exceptions

```python
from cookiecutter_modelfile.errors import (
    ModelNotFoundError,
    NetworkError,
    TemplateError,
    ValidationError,
    ConversionError
)
```

### Usage

```python
try:
    model = fetch_model("nonexistent/model")
except ModelNotFoundError as e:
    print(f"Model not found: {e}")
except NetworkError as e:
    print(f"Network error: {e}")
except TemplateError as e:
    print(f"Template error: {e}")
except ValidationError as e:
    print(f"Validation error: {e}")
except ConversionError as e:
    print(f"Conversion error: {e}")
```

---

## Configuration Models

### `HardwareProfile`

```python
@dataclass
class HardwareProfile:
    """Hardware hardware profile for a deployment target."""
    name: str
    devices: List[str]
    num_threads: int
    gpu_layers: int = -1
    vram: Optional[int] = None          # VRAM in bytes (int)
    min_context: Optional[int] = None   # minimum num_predict (int)
    description: str = ""
```

`HardwareProfile` is parsed by `load_config()` from `config.yml`. The `vram` field is stored in **bytes** and used to estimate `num_predict`. `min_context` enforces a lower bound on the estimated context length.

### `HardwareProfile.get_min_context()`

```python
def get_min_context(self, override: Optional[str] = None) -> Optional[int]:
    """Return the minimum context length for the given hardware override."""
    return self.get_hardware(override).min_context
```

### `HardwareProfile.print_config()`

```python
def print_config(self) -> None:
    """Print the hardware profile configuration to stdout."""
```

---

## Context Estimation

### `estimate_context()`

```python
from cookiecutter_modelfile.context import estimate_context

ctx = estimate_context(
    vram_bytes=10564246528,
    model_size_bytes=2970000000,
    tags=["quantization_type.Q4_K_M"],
    min_context=65536,
    vram_windows_overhead=vram_windows_overhead,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `vram_bytes` | int | Total GPU VRAM in bytes. `None` disables estimation. |
| `model_size_bytes` | int | GGUF model size in bytes. `None` disables estimation. |
| `tags` | List[str] | Optional model tags (e.g. `quantization_type.Q4_K_M`). |
| `min_context` | Optional[int] | Lower bound for the estimate. |
| `vram_windows_overhead` | Optional[int] | VRAM consumed by the OS on Windows. Defaults to `platform.system()`. |

**Return:** the estimated `num_predict` as an `int`, or `None` if VRAM/model size are missing or the usable VRAM is `<= 0`.

**Behavior:**
- `usable = vram_bytes - model_size_bytes - vram_windows_overhead`
- `context = int(usable / QUANT_BYTES_PER_TOKEN)` (quantization lookup via `tags`)
- If `min_context` is set and larger than the estimate, `context = min_context`
- On Windows, `vram_windows_overhead` defaults to `vram_windows_overhead` (~1.5 GB)

### Constants

```python
vram_windows_overhead = 1_570_000_000   # ~1.5GB Windows/WDDM overhead
DEFAULT_BYTES_PER_TOKEN = 4.0
VRAM_RESERVE = 0.25
```

---

## Version History

## Configuration

### Environment Variables

```bash
# Hugging Face API Token
HUGGING_FACE_HUB_TOKEN=your_token_here

# Custom API Base URL
HF_API_BASE=https://huggingface.co

# Debug Mode
COOKIE_CUTTER_DEBUG=1
```

### Pydantic Settings

```python
from cookiecutter_modelfile.config import Settings

settings = Settings()
print(settings.hf_api_token)
print(settings.debug_mode)
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-10-12 | Initial release |

---

*Generated automatically from source code*
