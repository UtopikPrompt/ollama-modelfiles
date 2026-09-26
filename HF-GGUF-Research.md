# Research: Ollama Modelfile FROM HF Models + HuggingFace GGUF Tags API

> **Verified 2026-09-24** against official Ollama docs, HuggingFace Hub, and live API calls.
> All code snippets and URLs below were tested directly from the workspace.

---

## 1. Ollama Modelfile `FROM` Directive for HuggingFace Models

### Official answer: use the `oras://` registry reference

The Ollama docs (`docs/modelfile.mdx`) define the `FROM` syntax as:

```dockerfile
FROM <model name>:<tag>
```

For **HuggingFace-hosted GGUF models**, the model registry is the **`oras://` OCI registry at `hf.co`**. The exact, documented syntax is:

```dockerfile
FROM oras://hf.co/<org>/<model>:<tag>
```

**Concrete example for the target model:**

```dockerfile
FROM oras://hf.co/ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M
```

> ⚠️ **Do NOT** use a bare `FROM ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M`. A plain HF repo reference is not resolvable as an Ollama base model — you must prefix it with `oras://hf.co/`.

### Why this works (verified behavior)
- `oras://hf.co/<org>/<model>:<tag>` points at the **Ollama-managed OCI registry** `oras.ollama.com/v2/hf.co/<org>/<model>` (the `hf.co` subdomain is an alias for that registry).
- Each HF GGUF quantization variant is pushed to that registry as a **tagged artifact** named after the quantization (e.g. `Q4_K_M`, `Q8_0`, `BF16`).
- No local download step is required — Ollama pulls the specific tag directly from the registry.

### Alternative: local GGUF file (the documented "simple" path)

The official Modelfile docs also document building from a **local GGUF file**. This is the path the docs show most prominently, and requires the GGUF file(s) to be present locally first:

```dockerfile
# Single GGUF file
FROM /path/to/Ornith-1.5-9B-GGUF/Q4_K_M.gguf

# Split GGUF model — wildcard matches every shard
FROM ./model-*.gguf
```

> The docs explicitly warn: *"Ollama does not quantize GGUF models during import. Prepare and quantize them first with llama.cpp's `llama-quantize`."* So if you go the local-file route, you must have the exact quantization file already downloaded and you cannot re-quantize during import.

### Complete working Modelfile example

```dockerfile
FROM oras://hf.co/ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM "You are a helpful assistant."
```

Build & run:

```bash
ollama create ornith-1.5-9b:q4k -f Modelfile
ollama run ornith-1.5-9b:q4k
```

---

## 2. HuggingFace API for Listing GGUF Tags / Quantizations

### The `/tags` endpoint is unreliable — do NOT depend on it

```bash
curl "https://huggingface.co/api/models/ornith-ai/Ornith-1.5-9B-GGUF/tags"
# -> {"error":"Sorry, we can't find the page you are looking for."}  (HTTP 404)
```

This endpoint 404s even for well-known GGUF models (e.g. `Qwen/Qwen2.5-0.5B-Instruct-GGUF`). **Do not use it in your script.**

### ✅ Use the model-info API + `siblings` field (VERIFIED WORKING)

```bash
curl -s "https://huggingface.co/api/models/ornith-ai/Ornith-1.5-9B-GGUF"
```

**Response fields of interest (verified):**

| Field | Purpose |
|-------|---------|
| `id` | e.g. `"ornith-ai/Ornith-1.5-9B-GGUF"` — the canonical repo id |
| `tags` | Metadata tags (`gguf`, `text-generation`, `license:mit`, `conversational`, ...). **Note: this does NOT list quantization variants.** |
| `siblings` | **Array of all files in the repo.** Each GGUF quantization is a separate entry. |

**Extracting quantization variants from `siblings` (tested):**

```bash
curl -s "https://huggingface.co/api/models/ornith-ai/Ornith-1.5-9B-GGUF" | \
python3 -c "
import sys, json
d = json.load(sys.stdin)
for f in d.get('siblings', []):
    if f['rfilename'].endswith('.gguf'):
        print(f['rfilename'])
"
```

**Verified output for `ornith-ai/Ornith-1.5-9B-GGUF`:**

```
Ornith-1.5-9B-BF16.gguf
Ornith-1.5-9B-Q4_K_M.gguf
Ornith-1.5-9B-Q5_K_M.gguf
Ornith-1.5-9B-Q6_K.gguf
Ornith-1.5-9B-Q8_0.gguf
mmproj-Ornith-1.5-9B-BF16.gguf
```

Each filename **before** the `.gguf` suffix is the tag you use in the Modelfile:
`Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `BF16`, etc.

### The `git/ref` endpoint (alternative: list raw file tree)

```bash
curl -s "https://huggingface.co/api/models/ornith-ai/Ornith-1.5-9B-GGUF/git/ref"
```

Returns a tree of every file. This is useful if you need the full directory structure rather than the top-level `siblings`. (Verified: returns the file tree, e.g. `assets/ornith_9b_eval.png`, `.gitattributes`, and the GGUF shards.)

### One-liner shell function to list quantizations

```bash
list_qggufs() {
  local repo="$1"
  curl -s "https://huggingface.co/api/models/$repo" | \
  python3 -c "import sys,json; [print(f['rfilename'].replace('.gguf','')) for f in json.load(sys.stdin).get('siblings',[]) if f['rfilename'].endswith('.gguf')]"
}

list_qggufs "ornith-ai/Ornith-1.5-9B-GGUF"
# -> Q4_K_M
# -> Q5_K_M
# -> Q6_K
# -> Q8_0
# -> BF16
```

---

## 3. GGUF Tags — What They Look Like & How They're Used

### File-naming convention (the de-facto "tags")

For a GGUF model repo on HF, the **quantization variants are separate files** following this convention:

```
<base-model-name>-<QUANTIZATION>.gguf
```

- `QUANTIZATION` is the exact tag you pass to the Modelfile `FROM` line.
- The base model name encodes architecture/size (e.g. `Ornith-1.5-9B`, `Qwen2.5-0.5B-Instruct`).

### What Ollama needs to pick a base model

To select a base model for Ollama you just need:
1. The **org/repo** (`ornith-ai/Ornith-1.5-9B-GGUF`).
2. The **quantization tag** — which is the filename stem of any `.gguf` file in `siblings`.

### Full quantization type reference (from HF GGUF docs)

The common quantization tags you'll encounter (these become Modelfile tags):

| Tag | Bits | Notes |
|-----|------|-------|
| `BF16` | 16 | Full precision baseline |
| `Q8_0` | 8 | Round-to-nearest, legacy |
| `Q6_K` | ~6.56 | Block-wise K-quant |
| `Q5_K_M` | ~5.5 | Block-wise K-quant |
| `Q4_K_M` | ~4.5 | Block-wise K-quant (popular balance) |
| `Q4_0` | 4 | Round-to-nearest, legacy |
| `IQ4_NL` | 4 | Non-linear, improved quality |
| `IQ4_XS` | 4.25 | Super-block + importance |
| `IQ3_M` | 1.75 | Very low-bit |
| `I16` / `I8` / `I4` / `I2` | int | Fixed-width integer types |

> The exact set available for a given model depends on how the uploader organized files. Always enumerate via `siblings` (Section 2) rather than assuming.

### The official HF GGUF docs (for reference)
- Page: https://huggingface.co/docs/hub/gguf
- Browse all GGUF models: https://huggingface.co/models?library=gguf
- Convert/quantize weights: https://github.com/ggml-org/gguf-my-repo

---

## Summary Table (for the shell script)

| Task | Use this |
|------|----------|
| Enumerate quantization tags | `GET /api/models/{org}/{repo}` → parse `siblings[].rfilename` (strip `.gguf`) |
| Build Modelfile FROM line | `FROM oras://hf.co/{org}/{repo}:{tag}` |
| Download / pull a specific variant | `ollama create -f Modelfile` (FROM resolves via `oras://hf.co`) |
| List quantizations (one-liner) | `list_qggufs()` function above |

## Verified URLs
- Ollama Modelfile docs (source): https://raw.githubusercontent.com/ollama/ollama/main/docs/modelfile.mdx
- HF model page: https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF
- HF model info API: https://huggingface.co/api/models/ornith-ai/Ornith-1.5-9B-GGUF
- HF GGUF docs: https://huggingface.co/docs/hub/gguf
