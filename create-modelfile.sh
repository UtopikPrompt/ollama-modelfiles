#!/bin/bash
set -e

# Configuration
MODELS_DIR="models"
DOWNLOADS_DIR="models"

# Default values for flags
FORCE_DOWNLOAD=false
NO_DOWNLOAD=false

# Parse command line arguments
while getopts "fnd" opt; do
    case $opt in
        f)
            FORCE_DOWNLOAD=true
            ;;
        n)
            NO_DOWNLOAD=true
            ;;
        d)
            show_usage
            exit 0
            ;;
        \?)
            show_usage
            exit 1
            ;;
    esac
done

show_usage() {
    echo "Usage: $0 [OPTIONS] <model_url>"
    echo "Options:"
    echo "  -f, --force    Force download even if file exists"
    echo "  -n, --no-download    Don't download any models"
    echo "  -d, --help     Show this help message"
}

# Load default parameters from config file if available
if [[ -f "config.bash" ]]; then
    source config.bash
fi

# Function to check if a parameter is active (not commented out) in config.bash
is_param_active() {
    local param_name="$1"
    [[ -f "config.bash" ]] || return 1
    grep -q "^#${param_name}" config.bash && return 1
    grep -q "^${param_name}=" config.bash && return 0
    return 1
}

# Function to get value of a parameter from config.bash
get_param_value() {
    local param_name="$1"
    [[ -f "config.bash" ]] || return 1
    # Clean up bash style evaluation to grab the actual variable assignment 
    # using indirect variable expansion ${!param_name} since config.bash was sourced
    echo "${!param_name:-}"
}

# Download and create Modelfile for each model
for model in "$@"; do
    echo "Downloading: $model"
    
    # Create directories if they don't exist
    mkdir -p "$MODELS_DIR"
    mkdir -p "$DOWNLOADS_DIR"
    
    # Extract organization from URL (e.g., "unsloth" from "huggingface.co/unsloth/Qwen3.5-4B-GGUF/...")
    organization=$(echo "$model" | awk -F/ '{print $(NF-4)}' | tr -d '\n' | head -c 50)
    
    # Extract model name from URL (handle filenames with spaces)
    model_name=$(echo "$model" | awk -F/ '{print $NF}' | sed 's/\.gguf$//' | tr -d '\n' | head -c 50)
    
    # Create organization subdirectory if it doesn't exist
    mkdir -p "$DOWNLOADS_DIR/$organization"
    
    # Check if file already exists
    DOWNLOAD_FILE="$DOWNLOADS_DIR/${organization}/${model_name}.gguf"
    if [[ -f "$DOWNLOAD_FILE" ]]; then
        if [[ "$FORCE_DOWNLOAD" == "true" ]]; then
            echo "  File exists, forcing download: $(basename "$model" .gguf)"
            curl -L "${model}" -o "$DOWNLOAD_FILE"
        else
            echo "  File already exists: $DOWNLOAD_FILE. Skipping download."
            if [[ "$NO_DOWNLOAD" != "true" ]]; then
                echo "  Skipping download (use -f to force)"
            fi
        fi
    else
        echo "  Downloading from Hugging Face: $(basename "$model" .gguf)"
        curl -L "${model}" -o "$DOWNLOAD_FILE"
    fi
    
    # Create organization subdirectory in models/ if it doesn't exist
    mkdir -p "$MODELS_DIR/$organization"

    modelfile_path="$MODELS_DIR/$organization/$model_name.Modelfile"

    # 1. Create the base template structure using a quoted Heredoc so nothing evaluates unexpectedly
    cat > "$modelfile_path" << 'EOF'
# 1. BASE MODEL (Required)
FROM ../../$DOWNLOADS_DIR/${organization}/${model_name}.gguf

# 2. PROMPT & CONVERSATION TEMPLATE
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
EOF

    # Fix the path string interpolation in the first line of the file since it was quoted
    sed -i "s|\$DOWNLOADS_DIR|${DOWNLOADS_DIR}|g" "$modelfile_path"
    sed -i "s|\${organization}|${organization}|g" "$modelfile_path"
    sed -i "s|\${model_name}|${model_name}|g" "$modelfile_path"

    # 2. Dynamically execute Bash checks and append the values safely to the Modelfile
    
    echo -e "\n# --- Model Behavior & Sampling ---" >> "$modelfile_path"
    is_param_active TEMPERATURE && echo "PARAMETER temperature      $(get_param_value TEMPERATURE)" >> "$modelfile_path"
    is_param_active TOP_K       && echo "PARAMETER top_k            $(get_param_value TOP_K)" >> "$modelfile_path"
    is_param_active TOP_P       && echo "PARAMETER top_p            $(get_param_value TOP_P)" >> "$modelfile_path"
    is_param_active MIN_P       && echo "PARAMETER min_p            $(get_param_value MIN_P)" >> "$modelfile_path"
    is_param_active SEED        && echo "PARAMETER seed             $(get_param_value SEED)" >> "$modelfile_path"

    echo -e "\n# --- Mirostat Perplexity Control (Alternative Sampling) ---" >> "$modelfile_path"
    is_param_active MIROSTAT     && echo "PARAMETER mirostat         $(get_param_value MIROSTAT)" >> "$modelfile_path"
    is_param_active MIROSTAT_ETA && echo "PARAMETER mirostat_eta     $(get_param_value MIROSTAT_ETA)" >> "$modelfile_path"
    is_param_active MIROSTAT_TAU && echo "PARAMETER mirostat_tau     $(get_param_value MIROSTAT_TAU)" >> "$modelfile_path"

    echo -e "\n# --- Context & Token Limits ---" >> "$modelfile_path"
    is_param_active NUM_CTX            && echo "PARAMETER num_ctx          $(get_param_value NUM_CTX)" >> "$modelfile_path"
    is_param_active NUM_PREDICT        && echo "PARAMETER num_predict      $(get_param_value NUM_PREDICT)" >> "$modelfile_path"
    is_param_active DRAFT_NUM_PREDICT  && echo "PARAMETER draft_num_predict $(get_param_value DRAFT_NUM_PREDICT)" >> "$modelfile_path"

    echo -e "\n# --- Penalties & Repetition ---" >> "$modelfile_path"
    is_param_active REPEAT_PENALTY   && echo "PARAMETER repeat_penalty   $(get_param_value REPEAT_PENALTY)" >> "$modelfile_path"
    is_param_active REPEAT_LAST_N    && echo "PARAMETER repeat_last_n    $(get_param_value REPEAT_LAST_N)" >> "$modelfile_path"
    is_param_active PRESENCE_PENALTY && echo "PARAMETER presence_penalty $(get_param_value PRESENCE_PENALTY)" >> "$modelfile_path"
    is_param_active FREQUENCY_PENALTY && echo "PARAMETER frequency_penalty $(get_param_value FREQUENCY_PENALTY)" >> "$modelfile_path"

    echo -e "\n# --- Hardware & Performance Knobs ---" >> "$modelfile_path"
    is_param_active NUM_KEEP       && echo "PARAMETER num_keep         $(get_param_value NUM_KEEP)" >> "$modelfile_path"
    is_param_active NUM_THREAD     && echo "PARAMETER num_thread       $(get_param_value NUM_THREAD)" >> "$modelfile_path"
    is_param_active NUM_GPU        && echo "PARAMETER num_gpu          $(get_param_value NUM_GPU)" >> "$modelfile_path"
    is_param_active MAIN_GPU       && echo "PARAMETER main_gpu         $(get_param_value MAIN_GPU)" >> "$modelfile_path"
    is_param_active NUMA           && echo "PARAMETER numa             $(get_param_value NUMA)" >> "$modelfile_path"
    is_param_active LOW_VRAM       && echo "PARAMETER low_vram         $(get_param_value LOW_VRAM)" >> "$modelfile_path"
    is_param_active F16_KV         && echo "PARAMETER f16_kv           $(get_param_value F16_KV)" >> "$modelfile_path"
    is_param_active VOCAB_ONLY     && echo "PARAMETER vocab_only       $(get_param_value VOCAB_ONLY)" >> "$modelfile_path"
    is_param_active USE_MMAP       && echo "PARAMETER use_mmap         $(get_param_value USE_MMAP)" >> "$modelfile_path"
    is_param_active USE_MLOCK      && echo "PARAMETER use_mlock        $(get_param_value USE_MLOCK)" >> "$modelfile_path"
    is_param_active EMBEDDING_ONLY && echo "PARAMETER embedding_only   $(get_param_value EMBEDDING_ONLY)" >> "$modelfile_path"

    echo "Created Modelfile: $modelfile_path"
    ollama create "$organization-$model_name" -f "$modelfile_path"
    echo "To update the model: ollama create \"$organization-$model_name\" -f \"$modelfile_path\""
done
