name: ollama-modelfile-generator
description: Complete automation for downloading GGUF models, creating optimized Modelfiles, and building Ollama models from Hugging Face URLs. Fully hands-off workflow.
instructions:
  # PHASE 1: Parse and Analyze URL
  - Analyze the provided Hugging Face GGUF URL.
  - Extract the organization name and model name from the URL path (e.g., ornith-ai/Ornith-1.5-9B-GGUF).
  - Extract the file information from the 'show_file_info' parameter.
  - Determine the model name and tag: Extract the model name and tag (e.g., Q4_K_M) from the 'show_file_info' parameter in the URL.
  - Construct the FROM directive: Format the directive as `huggingface.co/<organization>/<model_name>:<tag>`.
  
  # PHASE 2: Generate Optimized Modelfile
  - Generate a complete Ollama Modelfile with:
    - Corrected FROM directive using local path
    - All PARAMETER lines uncommented with hardware-optimized defaults:
      - PARAMETER num_ctx (default: 4096)
      - PARAMETER num_thread (default: 1)
      - PARAMETER num_gpu (default: 0)
      - PARAMETER num_gpu_mem (default: 4096 MiB)
      - PARAMETER num_batch (default: 16)
      - PARAMETER num_keep (default: 1000)
      - PARAMETER num_predict (default: -1, not "infinity")
      - PARAMETER num_keep_alive (default: 5m)
      - PARAMETER num_ngram (default: 1)
    - TEMPLATE with Llama-style tokens for better inference quality
    - SYSTEM message
    - LICENSE instruction
    - 2 MESSAGE examples for fine-tuning
    - REQUIRES instruction
  
  # PHASE 3: Download GGUF File
  - Download the GGUF file from Hugging Face to the workspace models directory.
  - Use curl with progress bar (-# flag) for efficient downloading.
  - Store file with proper naming: <organization>/<model_name>.gguf
  
  # PHASE 4: Create Modelfile
  - Write the complete Modelfile to: models/<organization>/<model_name>.Modelfile
  - Ensure file is created with correct permissions
  
  # PHASE 5: Create Ollama Model
  - Execute: ollama create <model_name> -f <modelfile_path>
  - Use --quiet flag to suppress verbose output
  - Wait for model creation to complete
  
  # PHASE 6: Verify Model
  - Execute: ollama show <model_name> --verbose
  - Confirm model exists with correct properties
  - Display model details to user
  
  # PHASE 7: Provide Summary
  - List all files created
  - Provide download size
  - Confirm model is ready to use
  - Include quick-start command

scope: comprehensive
examples:
  - url: https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF?show_file_info=Ornith-1.5-9B-Q4_K_M.gguf
  - hardware:desktop-steeve
  - workflow:
      steps:
        - downloaded: "Ornith-1.5-9B-GGUF.gguf"
        - modelfile: "models/ornith-ai/Ornith-1.5-9B.Modelfile"
        - ollama_model: "ornith-1.5-9B"
      model_details:
        quantization: Q4_K_M
        size: "~4.2 GB"
        format: "GGUF"
        ready: true
      quick_start: "ollama run ornith-1.5-9B"
      created_files:
        - models/ornith-ai/Ornith-1.5-9B-GGUF.gguf
        - models/ornith-ai/Ornith-1.5-9B.Modelfile
        - models/ornith-ai/Ornith-1.5-9B.safetensors (if applicable)