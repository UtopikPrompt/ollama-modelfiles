# Ollama Modelfiles 🦙

A centralized repository for storing and managing custom **Ollama Modelfiles**. These configurations optimize local Large Language Models (LLMs) for specific context lengths, system prompts, and operational parameters.

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have [Ollama](https://ollama.com) installed and running on your system.

### 2. Clone the Repository
```bash
git clone https://github.com
cd YOUR_REPO_NAME
```

### 3. Build a Model
To build a custom model from a `Modelfile` in this repository, run the `ollama create` command using a colon (`:`) to tag your model:

```bash
ollama create ornith-1.5:9b-64k -f ornith-1.5-9b-64k.modelfile
```

### 4. Run the Model
Once built, you can launch your custom model just like any official Ollama model:

```bash
ollama run ornith-1.5:9b-64k
```

---

## 📂 Available Modelfiles

| File Name | Base Model | Context Window | Purpose / Notes |
| :--- | :--- | :--- | :--- |
| `ornith-1.5-9b-64k.modelfile` | `ornith-1.5:9b` | **64,000 tokens** | Expanded context window for deep document analysis. |
| *(Add next file)* | | | |

---

## 🛠️ Modelfile Cheat Sheet

If you are adding new files to this repository, remember these core `Modelfile` parameters:

* `FROM` - Defines the base model (e.g., `FROM llama3.1` or `FROM ./local-model.gguf`).
* `PARAMETER num_ctx` - Sets the context window size in tokens (e.g., `64000`).
* `PARAMETER temperature` - Controls creativity/randomness (lower is more analytical, higher is more creative).
* `SYSTEM` - Injects a permanent system prompt to dictate the model's behavior or persona.

---

## 🔀 Managing Updates

If you modify a `.modelfile` in this repository, you must re-run the creation command to apply the changes to your local Ollama instance:

```bash
# Re-build after making edits
ollama create <model-tag-name> -f <filename>.modelfile
```
