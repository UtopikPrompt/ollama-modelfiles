from cookie_cutter_modelfile.utils import parse_huggingface_url

print("Testing URL parser:")
print("  URL -> Model ID:")
print(
    f'  {"https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"} -> {parse_huggingface_url("https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF")}'
)
print(
    f'  {"https://hf.co/model-something"} -> {parse_huggingface_url("https://hf.co/model-something")}'
)
print(
    f'  {"https://huggingface.co/api/models/org/model"} -> {parse_huggingface_url("https://huggingface.co/api/models/org/model")}'
)
print(
    f'  {"hf-coherence/command-r-plus"} -> {parse_huggingface_url("hf-coherence/command-r-plus")}'
)
print(f'  {"invalid-url"} -> {parse_huggingface_url("invalid-url")}')
print(f'  {""} -> {parse_huggingface_url("")}')
print(f"  {None} -> {parse_huggingface_url(None)}")
print("\nAll tests completed!")
