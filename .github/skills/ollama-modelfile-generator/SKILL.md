name: ollama-modelfile-generator
description: Automates creation of an Ollama Modelfile from a Hugging Face GGUF URL. Outputs the Modelfile content and the suggested path, using `FROM` and `PARAMETER` directives.
instructions:
  - Analyze the provided Hugging Face GGUF URL.
  - Extract the organization name and the file information (e.g., from the 'show_file_info' parameter).
  - Determine the filename: Use the `show_file_info` value if present and available; otherwise, use the model name from the URL. Strip the `.gguf` extension from the filename.
  - Construct the target Modelfile path using the format: `/models/<organization>/<filename>.Modelfile`.
  - Generate the Ollama Modelfile content, starting with the `FROM` directive pointing to the constructed GGUF path, followed by relevant `PARAMETER` directives (e.g., temperature, top_p).
  - Output the final Modelfile content and the suggested path to the user.
scope: quick
examples:
  - url: https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF?show_file_info=Ornith-1.5-9B-Q4_K_M.gguf
  - generated_output:
      path: /models/ornith-ai/Ornith-1.5-9B-Q4_K_M.Modelfile
      content: |
        FROM /models/ornith-ai/Ornith-1.5-9B-Q4_K_M.gguf
        # Ollama GGUF model loaded from Hugging Face
        PARAMETER temperature 0.8
        PARAMETER top_p 0.9
path: c:\Users\Steeve\Sources\github.com\SteeveGL\ollama-modelfiles\SKILL.md