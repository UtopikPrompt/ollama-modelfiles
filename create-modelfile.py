#!/usr/bin/env python3
"""
Ollama Modelfile Generator

Downloads GGUF model files from Hugging Face and creates Ollama Modelfiles
with configurable parameters.

Usage: python create-modelfile.py [OPTIONS] <model_url>
"""

import argparse
import os
import re
import subprocess
import sys
import urllib
from pathlib import Path
import yaml
from hf_hub import HFHubClient, ModelMetadata, fetch_model_metadata


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Ollama Modelfile Generator - Downloads GGUF model files from Hugging Face and creates Ollama Modelfiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://huggingface.co/Jackrong/Qwen3.5-4B-GGUF/main
  %(prog)s -f https://huggingface.co/Jackrong/Qwen3.5-4B-GGUF/main
  %(prog)s -n https://huggingface.co/Jackrong/Qwen3.5-4B-GGUF/main
        """
    )
    parser.add_argument(
        'model_url',
        type=str,
        help='Hugging Face model URL (e.g., https://huggingface.co/Jackrong/Qwen3.5-4B-GGUF/main)'
    )
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Force download even if file exists'
    )
    parser.add_argument(
        '-n', '--no-download',
        action='store_true',
        help="Don't download any models"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    return parser


class OllamaModelfileGenerator:
    def __init__(self, models_dir: str = "models", downloads_dir: str | None = None, *, config_dir: str | None = None, config_path: str | None = None, verbose: bool | None = None, env: bool | None = None):
        self.models_dir = Path(models_dir) if models_dir else Path("models")
        self.downloads_dir = Path(downloads_dir) if downloads_dir else Path("models")
        # config_dir is an alias for downloads_dir, provided for API compatibility
        self.config_dir = config_dir if config_dir else downloads_dir
        # config_path is the actual path to the config file (optional)
        self.config_path = config_path
        self.verbose = verbose if verbose is not None else False
        self.env = env if env is not None else False
        
    def show_usage(self):
        print("Usage: python create-modelfile.py [OPTIONS] <model_url>")
        print("Options:")
        print("  -f, --force    Force download even if file exists")
        print("  -n, --no-download    Don't download any models")
        print("  -h, --help     Show this help message")

    def load_config(self):
        """Load default parameters from config file if available (flat YAML format)"""
        config = {}
        config_file = self.downloads_dir / "config.yaml"
        if config_file.exists():
            with open(config_file, "r") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                # Directly copy flat YAML values to config
                for key, value in data.items():
                    if value is not None:
                        config[key] = str(value)
        # Fallback to config.bash for backward compatibility
        if not config:
            if (self.downloads_dir / "config.bash").exists():
                with open(self.downloads_dir / "config.bash", "r") as f:
                    content = f.read()
                # Extract parameter assignments (not commented out)
                for match in re.finditer(r'^(\w+)\s*=\s*"?([^"\s]+)"?', content, re.MULTILINE):
                    config[match.group(1)] = match.group(2)
        return config

    def is_param_active(self, param_name: str) -> bool:
        """Check if a parameter is active (not commented out) in config.yaml or config.bash"""
        config_file = self.downloads_dir / "config.yaml"
        if config_file.exists():
            with open(config_file, "r") as f:
                content = f.read()
            # Check if param is commented out in YAML (flat format)
            if re.search(rf'^#{param_name}\s*:', content, re.MULTILINE):
                return False
            # Check if param is assigned (not None or empty)
            return bool(re.search(rf'^{param_name}\s*:\s*[^#]', content, re.MULTILINE))
        # Fallback to config.bash
        if not (self.downloads_dir / "config.bash").exists():
            return False
        with open(self.downloads_dir / "config.bash", "r") as f:
            content = f.read()
        # Check if param is commented out
        if re.search(rf'^#{param_name}\s*=', content, re.MULTILINE):
            return False
        # Check if param is assigned
        return bool(re.search(rf'^{param_name}\s*=\s*"?[^"\s]+"?', content, re.MULTILINE))

    def get_param_value(self, param_name: str) -> str:
        """Get value of a parameter from config.yaml or config.bash"""
        config_file = self.downloads_dir / "config.yaml"
        if config_file.exists():
            with open(config_file, "r") as f:
                content = f.read()
            # Try to get value from YAML
            match = re.search(rf'^{param_name}\s*:\s*([^#]+)', content, re.MULTILINE)
            if match:
                value = match.group(1).strip()
                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                return value
        # Fallback to config.bash
        if not (self.downloads_dir / "config.bash").exists():
            return ""
        with open(self.downloads_dir / "config.bash", "r") as f:
            content = f.read()
        match = re.search(rf'^{param_name}\s*=\s*"?([^"\s]+)"?', content, re.MULTILINE)
        return match.group(1) if match else ""

    def download_file(self, url: str, dest_path: Path, force: bool = False) -> bool:
        """Download a file from URL to destination with progress bar"""
        try:
            if dest_path.exists() and not force:
                print(f"  ✓ File already exists: {dest_path}")
                return True
            
            # Extract hostname from URL for progress bar
            parsed_url = urllib.parse.urlparse(url)
            hostname = parsed_url.hostname or 'unknown'
            print(f"  ⬇ Downloading {dest_path.name} ({hostname})...")
            
            # Custom download function with progress bar
            def download_with_progress(block_num, block_size, total_size):
                downloaded = (block_num * block_size)
                if total_size:
                    percent = min(downloaded / total_size * 100, 100)
                    bar_length = 40
                    filled_length = int(bar_length * downloaded // total_size)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    sys.stdout.write(f'\r  [{bar}] {percent:.1f}%')
                    sys.stdout.flush()
                return block_num
            
            # Use urlretrieve with reporthook - NO manual while loop
            # The while loop was causing an infinite loop because we were reading
            # from urlopen AND calling urlretrieve simultaneously
            response = urllib.request.urlopen(url, timeout=10)
            total_size = int(response.headers.get('Content-Length', 0))
            urllib.request.urlretrieve(url, dest_path, reporthook=download_with_progress)
            response.close()
            
            print(f'\r  [████] 100% - Downloaded {total_size / 1024 / 1024:.2f} MB\n', flush=True)
            return True
            
        except urllib.error.HTTPError as e:
            print(f"  HTTP error downloading {url}: {e}")
            return False
        except urllib.error.URLError as e:
            print(f"  URL error downloading {url}: {e.reason}")
            return False
        except Exception as e:
            print(f"  Error downloading {url}: {e}")
            return False

    def download_split_files(self, base_url: str, downloads_dir: str) -> int:
        """Download all chunks of a split model file.
        
        Detects split files by pattern: -0000X-of-0000Y.gguf
        Downloads all chunks in order from base_url/resolve/main/
        Returns number of files downloaded.
        """
        import os
        
        # Base URL for resolve/main path
        base_path = "https://huggingface.co/resolve/main/"
        
        # Extract filename from URL (without extension)
        url_parts = base_url.split('/')
        if len(url_parts) < 6:
            print("  ✗ Invalid URL format for split model")
            return 0
        
        # Get the filename (e.g., "qwen2.5-coder-32b-instruct-q5_k_m-00001-of-00003.gguf")
        filename = url_parts[-1]
        
        # Extract chunk info: -00001-of-00003
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})\.', filename)
        if not chunk_match:
            print("  ✗ File is not a split file (no -X-of-Y pattern)")
            return 0
        
        chunk_num = int(chunk_match.group(1))
        total_chunks = int(chunk_match.group(2))
        
        # Construct base URL for this chunk
        chunk_url = f"{base_path}{'/'.join(url_parts[:-1])}/{filename}"
        
        # Determine base filename (without chunk suffix)
        base_filename = filename
        if chunk_match:
            # Remove the chunk suffix
            base_filename = filename[:chunk_match.start()] + '.gguf'
        
        # Calculate total size for progress bar
        total_size = total_chunks * (int(chunk_url.split('/')[-2].split('-')[2]) if '-' in url_parts[-2] else 0)
        
        print(f"  Found split model: {total_chunks} chunks")
        print(f"  Chunk {chunk_num}/{total_chunks}")
        
        # Download all chunks in order
        for i in range(1, total_chunks + 1):
            chunk_file = Path(downloads_dir) / base_filename
            chunk_file.chmod(0o755)  # Fix permissions
            
            # Construct URL for this chunk
            chunk_file_name = f"{base_filename}-{i}.gguf"
            chunk_url_full = f"{base_path}{'/'.join(url_parts[:-1])}/{chunk_file_name}"
            chunk_dest = Path(downloads_dir) / chunk_file_name
            chunk_dest.chmod(0o755)  # Fix permissions
            
            print(f"  ⬇ Chunk {i}/{total_chunks}")
            
            if self.download_file(chunk_url_full, chunk_dest):
                print(f"  ✓ Chunk {i}/{total_chunks} downloaded")
                
                # Progress bar for all chunks
                downloaded = i
                if total_size:
                    progress = min(downloaded / total_size * 100, 100)
                    bar_length = 40
                    filled_length = int(bar_length * downloaded // total_size)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    sys.stdout.write(f'\r  [{bar}] {progress:.1f}% ({i}/{total_chunks})')
                    sys.stdout.flush()
            else:
                print(f"  ✗ Failed to download chunk {i}/{total_chunks}")
                return 0
        
        print(f'\r  [████] 100% - Downloaded {total_size / 1024 / 1024:.2f} MB (all {total_chunks} chunks)\n', flush=True)
        return total_chunks

    def fetch_metadata(self, model_url: str) -> ModelMetadata | None:
        """Fetch metadata for a model from Hugging Face Hub.
        
        Args:
            model_url: URL of the model (e.g., "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/main")
            
        Returns:
            ModelMetadata object if successful, None otherwise
        """
        try:
            # Extract repo ID from URL (e.g., "unsloth/Qwen3.5-4B-GGUF")
            repo_id = self.extract_organization(model_url)
            if not repo_id:
                print("  ✗ Could not extract organization from URL")
                return None
            
            # Extract model name from URL (e.g., "Qwen3.5-4B-GGUF")
            model_name = self.extract_model_name(model_url)
            if not model_name:
                print("  ✗ Could not extract model name from URL")
                return None
            
            # Construct HF Hub URL
            hf_url = f"https://huggingface.co/api/models/{repo_id}/{model_name}"
            
            # Fetch metadata using HFHubClient
            metadata = fetch_model_metadata(hf_url)
            if metadata:
                print(f"  ✓ Metadata fetched: {metadata.name} ({metadata.size_in_gb:.2f} GB)")
            return metadata
            
        except Exception as e:
            print(f"  ⚠ Could not fetch metadata: {e}")
            return None

    def generate_modelfile_name(self, model_url: str, metadata: ModelMetadata | None = None) -> str:
        """Generate an enhanced modelfile name using metadata.
        
        If metadata is available, uses format: {model_name}-{size_in_gb}-{quantization}
        Otherwise, falls back to: {organization}-{model_name}
        
        Args:
            model_url: URL of the model
            metadata: Optional metadata for enhanced naming
            
        Returns:
            Enhanced modelfile name
        """
        # Extract base organization and model name
        organization = self.extract_organization(model_url)
        model_name = self.extract_model_name(model_url)
        
        # Try to extract quantization from model name
        quantization = ""
        for q in ["Q4_K_M", "Q4_0", "Q5_K_M", "Q5_0", "Q6_K", "Q6_K_M", 
                  "Q8_0", "Q8_1", "Q8_K_M", "Q2_K", "Q3_K_M", "Q4_0", 
                  "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "Q8_1"]:
            if q in model_name:
                quantization = q
                break
        
        # Use metadata for enhanced naming if available
        if metadata:
            size_gb = metadata.size_in_gb
            name_parts = [model_name]  # Preserve full model name including chunk suffix
            if size_gb:
                name_parts.append(f"{size_gb:.1f}GB")
            if quantization:
                name_parts.append(quantization)
            return "-".join(name_parts)
        
        # Fallback naming
        return f"{organization}-{model_name}"

    def extract_organization(self, model_url: str) -> str:
        """Extract organization from URL (e.g., "unsloth" from "huggingface.co/unsloth/Qwen3.5-4B-GGUF/...")"""
        # Match pattern: huggingface.co/{org}/{repo}/...
        match = re.search(r'huggingface\.co/([^/]+)/([^/]+)/', model_url)
        if match:
            return match.group(1)
        # Fallback: try to extract from path
        parts = model_url.split("/")
        if len(parts) >= 4:
            return parts[2][:50]  # Limit to 50 chars
        return ""

    def extract_model_name(self, model_url: str) -> str:
        """Extract model name from URL (handle filenames with spaces).
        
        For split models, strips the chunk suffix to get the base name.
        e.g., "Qwen3.5-4B-GGUF-00006.gguf" -> "Qwen3.5-4B-GGUF"
        """
        # Get filename without extension
        filename = Path(model_url).name
        model_name = filename.rsplit(".", 1)[0] if "." in filename else filename
        
        # For split models, remove chunk suffix
        # Pattern: -XXXXX-of-XXXXX (e.g., -00001-of-00003)
        chunk_match = re.search(r'-\d{5}-of-\d{5}$', model_name)
        if chunk_match:
            model_name = model_name[:chunk_match.start()]
        
        # Limit to 50 chars
        return model_name[:50]

    def create_base_modelfile(self, modelfile_path: Path, organization: str, model_name: str, downloads_dir: str):
        """Create the base template structure using a quoted Heredoc so nothing evaluates unexpectedly"""
        heredoc_content = """# 1. BASE MODEL (Required)
FROM ../../${downloads_dir}/${organization}/${model_name}.gguf

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
# =============================================================================="""

        # Replace variables after creating heredoc
        heredoc_content = heredoc_content.replace("${downloads_dir}", downloads_dir)
        heredoc_content = heredoc_content.replace("${organization}", organization)
        heredoc_content = heredoc_content.replace("${model_name}", model_name)

        with open(modelfile_path, "w") as f:
            f.write(heredoc_content)

    def create_modelfile(self, model_url: str, organization: str, model_name: str, downloads_dir: str, metadata: ModelMetadata | None = None):
        """Download model and create Modelfile with parameters
        
        Args:
            model_url: URL of the model
            organization: Organization name from URL
            model_name: Model name from URL
            downloads_dir: Path to downloads directory
            metadata: Optional Hugging Face metadata for enhanced naming
        """
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        (self.downloads_dir / organization).mkdir(parents=True, exist_ok=True)
        (self.models_dir / organization).mkdir(parents=True, exist_ok=True)

        # Paths
        # Extract base filename without chunk suffix for split models
        # e.g., "qwen2.5-coder-32b-instruct-q5_k_m-00001-of-00003" -> "qwen2.5-coder-32b-instruct-q5_k_m"
        base_model_name = model_name
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})$', model_name)
        if chunk_match:
            # Remove the chunk suffix to get the base filename
            base_model_name = model_name[:chunk_match.start()]
        
        download_file = self.downloads_dir / organization / f"{base_model_name}.gguf"
        # Use metadata for enhanced modelfile naming
        modelfile_name = metadata.generate_modelfile_name(model_name) if hasattr(metadata, 'generate_modelfile_name') else base_model_name
        modelfile_path = self.models_dir / organization / f"{modelfile_name}.Modelfile"

        # 1. Create the base template structure
        self.create_base_modelfile(modelfile_path, organization, model_name, downloads_dir)

        # 2. Download the model with progress bar
        # Use base_model_name for "file already exists" check since chunks are stored with base names
        base_model_path = self.downloads_dir / organization / f"{base_model_name}.gguf"
        # Check if ALL chunk files exist (for split models) or base file (for regular models)
        # Chunk files have format: {base_model_name}-{XXXXX}.gguf
        chunk_exists = False
        if chunk_match:
            # For split models, check if ALL chunk files exist
            # Chunk files are named: {base_model_name}-{00001}.gguf, {base_model_name}-{00002}.gguf, etc.
            # We must verify EVERY chunk exists, not just the first one
            all_chunks_exist = True
            for i in range(1, int(chunk_match.group(2)) + 1):
                chunk_path = self.downloads_dir / organization / f"{base_model_name}-{i:05d}.gguf"
                if not chunk_path.exists():
                    all_chunks_exist = False
                    break  # Stop early if any chunk is missing
            chunk_exists = all_chunks_exist
        else:
            # For regular models, check if base file exists
            chunk_exists = base_model_path.exists()
        
        if chunk_exists:
            print(f"  ✓ Model already exists: {download_file}")
            return  # File exists, skip download entirely

        # 3. Append parameters dynamically
        config = self.load_config()

        # --- Model Behavior & Sampling ---
        print("\n# --- Model Behavior & Sampling ---")
        for param in ["TEMPERATURE", "TOP_K", "TOP_P", "MIN_P", "SEED"]:
            if self.is_param_active(param):
                value = self.get_param_value(param)
                print(f"  Adding {param} = {value}")
                with open(modelfile_path, "a") as f:
                    f.write(f"PARAMETER {param} {value}\n")

        # --- Mirostat Perplexity Control (Alternative Sampling) ---
        print("\n# --- Mirostat Perplexity Control (Alternative Sampling) ---")
        for param in ["MIROSTAT", "MIROSTAT_ETA", "MIROSTAT_TAU"]:
            if self.is_param_active(param):
                value = self.get_param_value(param)
                print(f"  Adding {param} = {value}")
                with open(modelfile_path, "a") as f:
                    f.write(f"PARAMETER {param} {value}\n")

        # --- Context & Token Limits ---
        print("\n# --- Context & Token Limits ---")
        for param in ["NUM_CTX", "NUM_PREDICT", "DRAFT_NUM_PREDICT"]:
            if self.is_param_active(param):
                value = self.get_param_value(param)
                print(f"  Adding {param} = {value}")
                with open(modelfile_path, "a") as f:
                    f.write(f"PARAMETER {param} {value}\n")

        # --- Penalties & Repetition ---
        print("\n# --- Penalties & Repetition ---")
        for param in ["REPEAT_PENALTY", "REPEAT_LAST_N", "PRESENCE_PENALTY", "FREQUENCY_PENALTY"]:
            if self.is_param_active(param):
                value = self.get_param_value(param)
                print(f"  Adding {param} = {value}")
                with open(modelfile_path, "a") as f:
                    f.write(f"PARAMETER {param} {value}\n")

        # --- Hardware & Performance Knobs ---
        print("\n# --- Hardware & Performance Knobs ---")
        for param in ["NUM_KEEP", "NUM_THREAD", "NUM_GPU", "MAIN_GPU", "NUMA", "LOW_VRAM", 
                      "F16_KV", "VOCAB_ONLY", "USE_MMAP", "USE_MLOCK", "EMBEDDING_ONLY"]:
            if self.is_param_active(param):
                value = self.get_param_value(param)
                print(f"  Adding {param} = {value}")
                with open(modelfile_path, "a") as f:
                    f.write(f"PARAMETER {param} {value}\n")

        print(f"\nCreated Modelfile: {modelfile_path}")
        print(f"To update the model: ollama create \"{organization}-{model_name}\" -f \"{modelfile_path}\"")

    def process_model(self, model_url: str, force: bool = False, no_download: bool = False):
        """Process a single model"""
        print(f"\nProcessing: {model_url}")

        organization = self.extract_organization(model_url)
        model_name = self.extract_model_name(model_url)

        print(f"  Organization: {organization}")
        print(f"  Model name: {model_name}")

        # Fetch metadata for enhanced naming
        metadata = self.fetch_metadata(model_url)

        # Create organization subdirectory in downloads if it doesn't exist
        (self.downloads_dir / organization).mkdir(parents=True, exist_ok=True)

        # Check if file already exists
        download_file = self.downloads_dir / organization / f"{model_name}.gguf"
        if download_file.exists():
            if force:
                print(f"  File exists, forcing download: {model_name}")
                self.download_file(model_url, download_file, force=True)
            elif no_download:
                print(f"  File already exists: {download_file}. Skipping download.")
                print(f"  Skipping download (use -f to force)")
            else:
                print(f"  File already exists: {download_file}. Skipping download.")
        else:
            print(f"  Downloading from Hugging Face: {model_name}")
            if self.download_file(model_url, download_file, force=force):
                print(f"  Download complete: {download_file}")
            else:
                print(f"  Failed to download model. Aborting.")
                return

        # Create organization subdirectory in models/ if it doesn't exist
        (self.models_dir / organization).mkdir(parents=True, exist_ok=True)

        # Generate modelfile name - use metadata for enhanced naming
        modelfile_name = self.generate_modelfile_name(model_url, metadata)
        print(f"  Modelfile name: {modelfile_name}")

        # Pass downloads_dir last, then modelfile_name
        self.create_modelfile(model_url, organization, model_name, str(self.downloads_dir), modelfile_name)
        print("\nOllama Modelfile Generator - Downloads GGUF models and creates Modelfiles")

    @staticmethod
    def create_parser():
        """Create and return the argument parser"""
        parser = argparse.ArgumentParser(
            description="Ollama Modelfile Generator - Downloads GGUF models and creates Modelfiles",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF/main
  %(prog)s https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF/main -n my-model
  %(prog)s https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF/main -d ./custom-downloads
  %(prog)s https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF/main --no-download
  %(prog)s https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF/main --verbose

Environment Variables:
  OLLAMA_DOWNLOADS_DIR    Path to downloads directory (default: ./downloads)
  OLLAMA_MODELS_DIR       Path to models directory (default: ./models)
  OLLAMA_CONFIG_DIR       Path to config directory (default: ./downloads)
  OLLAMA_NO_DOWNLOAD      Set to '1' to skip all downloads
"""
        )

        # Main positional argument
        parser.add_argument(
            "url",
            type=str,
            help="Hugging Face model URL to process (e.g., https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF/main)"
        )

        # Download options
        parser.add_argument(
            "-n", "--name",
            type=str,
            default=None,
            help="Custom name for the modelfile (overrides auto-generated name)"
        )
        parser.add_argument(
            "-d", "--downloads-dir",
            type=str,
            default=None,
            help="Path to downloads directory (default: ./downloads)"
        )
        parser.add_argument(
            "-m", "--models-dir",
            type=str,
            default=None,
            help="Path to models directory (default: ./models)"
        )
        parser.add_argument(
            "--no-download",
            action="store_true",
            help="Skip model download, only create modelfile from existing file"
        )

        # Configuration options
        parser.add_argument(
            "-c", "--config",
            type=str,
            default=None,
            help="Path to config.yaml file"
        )
        parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Enable verbose output"
        )
        parser.add_argument(
            "--config-dir",
            type=str,
            default=None,
            help="Path to config directory (default: ./downloads)"
        )

        # Environment variable overrides
        parser.add_argument(
            "--env",
            action="store_true",
            help="Load all environment variables as config"
        )

        # Help is handled by argparse automatically
        # parser.add_argument(
        #     "-h", "--help",
        #     action="help",
        #     help="Show this help message and exit"
        # )

        return parser


def main():
    """Main entry point"""
    # Create argument parser
    parser = OllamaModelfileGenerator.create_parser()

    # Parse arguments
    args = parser.parse_args()

    # Validate required arguments
    if not args.url:
        parser.print_help()
        print("\nError: URL is required")
        print("Usage: %(prog)s <huggingface-model-url>")
        exit(1)

    # Create generator instance
    generator = OllamaModelfileGenerator(
        downloads_dir=args.downloads_dir,
        models_dir=args.models_dir,
        config_dir=args.config_dir,
        config_path=args.config,
        verbose=args.verbose,
        env=args.env
    )

    # Process model
    generator.process_model(args.url, force=False, no_download=args.no_download)

    return 0


if __name__ == "__main__":
    exit(main())
