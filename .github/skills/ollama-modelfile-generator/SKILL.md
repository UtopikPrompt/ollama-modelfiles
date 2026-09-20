name: ollama-modelfile-generator
description: Automates creation of Ollama Modelfiles from Hugging Face GGUF URLs, generating multiple versions adapted to different hardware configurations (CPU, RAM, GPU, VRAM).
instructions:
  - Read hardware configurations from config.yaml
  - Analyze the provided Hugging Face GGUF URL.
  - Extract the organization name and file information (e.g., from the 'show_file_info' parameter).
  - Determine the model name and tag: Extract the model name and tag (e.g., Q4_K_M) from the 'show_file_info' parameter in the URL.
  - Detect user's hardware configuration by analyzing system info or user input
  - Match the model size to appropriate hardware configuration
  - Construct the FROM directive: Format the directive as `huggingface.co/<organization>/<model_name>:<tag>`.
  - Generate multiple Ollama Modelfile versions:
    - Primary Modelfile (recommended configuration)
    - Alternative Modelfiles (compatible configurations)
  - Include hardware-specific PARAMETER directives:
    - PARAMETER num_ctx (context window based on RAM)
    - PARAMETER num_thread (CPU cores)
    - PARAMETER num_gpu (GPU count)
    - PARAMETER num_gpu_mem (VRAM allocation)
    - PARAMETER num_batch (batch size based on RAM)
    - PARAMETER keep_alive (keep time based on use case)
  - Output quantization recommendations based on model size and available VRAM
  - Output the final Modelfile content and suggested path to the user
scope: quick
examples:
  - url: https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF?show_file_info=Ornith-1.5-9B-Q4_K_M.gguf
  - hardware: low-end-desktop
  - generated_output:
      primary_path: /models/ornith-ai/Ornith-1.5-9B.Modelfile
      primary_content: |
        FROM huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M
        # Ollama GGUF model loaded from Hugging Face
        # Recommended for: Low-end Desktop (4 CPU, 8GB RAM, no GPU)
        PARAMETER num_ctx 2048
        PARAMETER num_thread 4
        PARAMETER keep_alive 5m
      alternative_paths:
        - /models/ornith-ai/Ornith-1.5-9B-Essential.Modelfile
        - /models/ornith-ai/Ornith-1.5-9B-Basic.Modelfile
      quantization_recommendation: Q4_K_M (4-bit quantization optimal for 1.5B models)
path: c:\Users\Steeve\Sources\github.com\SteeveGL\ollama-modelfiles\config.yaml