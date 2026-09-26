#!/bin/bash
set -e

# Configuration
MODELS_DIR="models"
DOWNLOADS_DIR="downloads"

# Download and create Modelfile for each model
for model in "$@"; do
    echo "Downloading: $model"
    
    # Create directories if they don't exist
    mkdir -p "$MODELS_DIR"
    mkdir -p "$DOWNLOADS_DIR"
    
    # Extract organization from URL (e.g., "unsloth" from "huggingface.co/unsloth/Qwen3.5-4B-GGUF/...")
    organization=$(echo "$model" | awk -F/ '{print $(NF-4)}')
    
    # Extract model name from URL (handle filenames with spaces)
    model_name=$(echo "$model" | awk -F/ '{print $NF}' | sed 's/\.gguf$//')
    
    # Create organization subdirectory if it doesn't exist
    mkdir -p "$DOWNLOADS_DIR/$organization"
    
    # Download the model first (from Hugging Face)
    echo "  Downloading from Hugging Face: $(basename "$model" .gguf)"
    curl -L "${model}" -o "$DOWNLOADS_DIR/${organization}/${model_name}.gguf"
    
    # Create organization subdirectory in models/ if it doesn't exist
    mkdir -p "$MODELS_DIR/$organization"

    # Create Modelfile in models/ directory
    cat > "$MODELS_DIR/$organization/$model_name.Modelfile" << EOF
# 1. BASE MODEL (Required)
FROM "$DOWNLOADS_DIR/${organization}/${model_name}.gguf"

# 2. PROMPT & CONVERSATION TEMPLATE
# Defines how system messages, prompt strings, and model outputs are structured.
#TEMPLATE """
#{{- if .System }}<|start_header_id|>system<|end_header_id|>
#{{ .System }}<|eot_id|>
#{{- end }}
#{{- if .Prompt }}<|start_header_id|>user<|end_header_id|>
#{{ .Prompt }}<|eot_id|>
#{{- end }}<|start_header_id|>assistant<|end_header_id|>
#{{ .Response }}<|eot_id|>"""

## 3. SYSTEM MESSAGES & PERSONA
#SYSTEM """You are a highly helpful, precise local AI assistant."""

## 4. PRE-LOADED CONVERSATION HISTORY (Few-Shot Prompting Examples)
#MESSAGE user "Hello! What is your purpose?"
#MESSAGE assistant "I am configured via a custom Modelfile to assist you locally."

# ==============================================================================
# RUNTIME GENERATION PARAMETERS
# ==============================================================================

# --- Model Behavior & Sampling ---
#PARAMETER temperature      0.8    # Creativity control (0.0 = strict/deterministic, 1.0+ = highly creative)
#PARAMETER top_k            40     # Caps the generation pool to the top K most likely tokens
#PARAMETER top_p            0.9    # Nucleus sampling threshold (filters out low-probability choices)
#PARAMETER min_p            0.0    # Minimum probability threshold relative to the top token
#PARAMETER seed             0      # Random number seed (set an integer > 0 for reproducible outputs)

# --- Mirostat Perplexity Control (Alternative Sampling) ---
#PARAMETER mirostat         0      # Enable Mirostat (0 = disabled, 1 = Mirostat 1.0, 2 = Mirostat 2.0)
#PARAMETER mirostat_eta     0.1    # Learning rate/responsiveness adjustment for Mirostat
#PARAMETER mirostat_tau     5.0    # Balancing parameter for text coherence vs diversity

# --- Context & Token Limits ---
#PARAMETER num_ctx          2048   # Maximum text history the model remembers (e.g., 4096, 8192)
#PARAMETER num_predict      -1     # Max tokens to generate per response (-1 = infinite/until stop sequence)
#PARAMETER draft_num_predict 4     # Speculative draft tokens to predict per step (if using draft models)

# --- Penalties & Repetition ---
#PARAMETER repeat_penalty   1.1    # How aggressively to block word and phrase repetition
#PARAMETER repeat_last_n    64     # How far back (tokens) to scan for text duplication (0 = off, -1 = context)
#PARAMETER presence_penalty 0.0    # Degree to penalize tokens that have already appeared in output
#PARAMETER frequency_penalty 0.0   # Degree to penalize tokens based on cumulative usage frequency

# --- Hardware & Performance Knobs ---
#PARAMETER num_keep         4      # Amount of original prompt tokens to anchor if context shifts
#PARAMETER num_thread       8      # CPU compute threads (Ollama auto-sets this by default; set explicitly if needed)
#PARAMETER num_gpu          99     # Model layers to drop into VRAM (0 = CPU only, 99 = push everything to GPU)
#PARAMETER main_gpu         0      # Sets target primary GPU ID when running a multi-GPU environment
#PARAMETER numa             false  # Toggles Non-Uniform Memory Access balancing tweaks
#PARAMETER low_vram         false  # Streamlines internal layers to fit strict resource ceilings
#PARAMETER f16_kv           true   # Retains half-precision structures for Key/Value generation caches
#PARAMETER vocab_only       false  # Instructs Ollama to only load dictionary structures (omits weight layers)
#PARAMETER use_mmap         true   # Leverages file mapping memory strategies to speed up startup
#PARAMETER use_mlock        false  # Permanently binds layers to active RAM allocations (avoids system disk swapping)
#PARAMETER embedding_only   false  # Silences normal output workflows to only act as an embedding extractor

# --- Stop Sequences ---
## Declares exact sequences that immediately cut off further text generation.
#PARAMETER stop "<|eot_id|>"
#PARAMETER stop "User:"
#PARAMETER stop "Assistant:"
EOF

    ollama create "$organization/$model_name" -f "$MODELS_DIR/$organization/$model_name.Modelfile"

    echo "Created Modelfile: $MODELS_DIR/$organization/$model_name.Modelfile"
done
