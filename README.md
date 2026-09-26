# Ollama Modelfiles 🦙

A centralized repository for storing and managing custom **Ollama Modelfiles**. These configurations optimize local Large Language Models (LLMs) for specific context lengths, system prompts, and operational parameters.

## � Directory Structure

```
ollama-modelfiles/
├── downloads/
├── models/
├── config.bash              # Active configuration
├── config-example.bash      # Configuration template
├── create-modelfile.sh      # Modelfile generation script
├── README.md                # This file
└── HF-GGUF-Research.md     # Research documentation
```

## ⚙️ Configuration

### Active Configuration (`config.bash`)```bash
# Sampling Parameters
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=40
REPEAT_PENALTY=1.5
REPEAT_LAST_N=64

# Context & Token Limits
NUM_CTX=65536

# Hardware Parameters
NUM_THREAD=4
```

### Configuration Template (`config-example.bash`)```bash
# System prompt
#SYSTEM="You are a helpful assistant."

# Sampling Parameters
#TEMPERATURE=0.7
#TOP_P=0.9
#TOP_K=40
#REPEAT_PENALTY=1.5
#REPEAT_LAST_N=64
#REPEAT_PENALTY_PREFIX=1.1
#MIN_P=0.1
#MINING_P=0.5
#SEED=42

# Mirostat Parameters
#MIROSTAT=0
#MIROSTAT_TAU=5.0
#MIROSTAT_ETA=0.1
#PENALIZE_NEW_TOKENS=0.5

# Context & Token Limits
#NUM_PREDICT=4096
NUM_CTX=65536
#NUM_BATCH=512

# Hardware Parameters
#NUM_THREAD=4
#NUM_GPU=-1
#SWAP_SPACE=4

# Additional Parameters
#NUM_KEEP=4
#MAIN_GPU=0
#NUMA="false"
#LOW_VRAM="false"
#F16_KV="true"
#VOCAB_ONLY="false"
#USE_MMAP="true"
#USE_MLOCK="false"
#EMBEDDING_ONLY="false"

# Stop Sequences
#STOP_1="<|eot_id|>"
#STOP_2="User:"
#STOP_3="Assistant:"
```

### Automated Modelfile Generation

Use the `create-modelfile.sh` script to generate Modelfiles from GGUF files:

```bash
# Generate Modelfile for a specific model
./create-modelfile.sh -f ../../downloads/Qwen/qwen2.5-coder-7b-instruct-q8_0.gguf

# Generate Modelfile with a custom name
./create-modelfile.sh -n my-custom-model -f ../../downloads/Qwen/qwen2.5-coder-7b-instruct-q8_0.gguf

# Generate Modelfile and save to specific location
./create-modelfile.sh -d ./my-location -f ../../downloads/Qwen/qwen2.5-coder-7b-instruct-q8_0.gguf
```

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have [Ollama](https://ollama.com) installed and running on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/UtopikPrompt/ollama-modelfiles.git
cd ollama-modelfiles
```

### 3. Build a Model
To build a custom model from a `Modelfile` in this repository, run the `ollama create` command using a colon (`:`) to tag your model:

```bash
ollama create qwen-7b-64k:custom -f models/Qwen/qwen2.5-coder-7b-instruct-q8_0.Modelfile
```

### 4. Run the Model
Once built, you can launch your custom model just like any official Ollama model:

```bash
ollama run qwen-7b-64k:custom
```

## 🛠️ Modelfile Parameters Reference

### Core Parameters
| Parameter | Description |
|-----------|-------------|
| `FROM` | Defines the base model (GGUF file path) |
| `PARAMETER num_ctx` | Sets the context window size in tokens (default: 4096) |
| `PARAMETER num_predict` | Maximum number of tokens to generate (default: 2048) |
| `PARAMETER temperature` | Controls randomness (0.0 = deterministic, 1.0 = random) |
| `PARAMETER top_p` | Nucleus sampling threshold (0.0 to 1.0) |
| `PARAMETER top_k` | Top-k sampling threshold (1 to 100) |
| `PARAMETER repeat_penalty` | Penalizes repeated tokens (1.0 = no penalty) |
| `PARAMETER repeat_last_n` | Context size for repeat penalty |

### System & Template
| Parameter | Description |
|-----------|-------------|
| `SYSTEM` | Persistent system prompt injected into every request |
| `TEMPLATE` | Conversation template for formatting prompts |
| `MESSAGE` | Pre-loaded conversation history |

### Hardware & Performance
| Parameter | Description |
|-----------|-------------|
| `PARAMETER num_thread` | Number of CPU threads for processing |
| `PARAMETER num_gpu` | Number of GPUs to use for offloading |
| `PARAMETER main_gpu` | Primary GPU index for offloading |
| `PARAMETER low_vram` | Enable low VRAM mode |
| `PARAMETER f16_kv` | Use FP16 for KV cache |
| `PARAMETER vocab_only` | Only load vocabulary (memory efficient) |
| `PARAMETER use_mmap` | Use memory mapping for file access |
| `PARAMETER use_mlock` | Lock memory in RAM (prevents swapping) |

## 🔀 Managing Updates

If you modify a `.modelfile` in this repository, you must re-run the creation command to apply the changes to your local Ollama instance:

```bash
# Re-build after making edits
ollama create <model-tag-name> -f <filename>.modelfile

# Remove old model
ollama rm <model-tag-name>
```

## 📚 Documentation

- **[config-example.bash](config-example.bash)** - Configuration template with all options
- **[create-modelfile.sh](create-modelfile.sh)** - Automated Modelfile generation script