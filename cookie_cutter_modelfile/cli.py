"""
Command-line interface for the cookie cutter modelfile tool.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .converter import OllamaConverter
from .huggingface import HFClient
from .modelfile import ModelfileGenerator
from .templates import TemplateManager
from .utils import (
    get_model_info_display,
    normalize_model_id,
    parse_huggingface_url,
    print_error,
)


class CLI:
    """Command-line interface for the modelfile tool."""

    def __init__(self):
        """Initialize the CLI."""
        self.console = Console()
        self.hf_client = HFClient()
        self.template_manager = TemplateManager()
        self.modelfile_generator = ModelfileGenerator()
        self.ollama_converter = OllamaConverter()

    def _parse_model_id(self, model_id: str) -> Optional[str]:
        """Parse and normalize a model ID, handling URLs.

        Args:
            model_id: The model identifier (can be a URL or plain ID).

        Returns:
            The normalized model ID, or None if parsing fails.
        """
        # Try to parse as Hugging Face URL first
        parsed_model_id = parse_huggingface_url(model_id)
        if parsed_model_id:
            self.console.print("[bold yellow]Parsed URL to model ID:[/bold yellow]")
            self.console.print(f"   {model_id} -> {parsed_model_id}")
            return parsed_model_id

        # Fall back to normalize_model_id for plain IDs
        return normalize_model_id(model_id)

    def create(self, model_id: str, output: Optional[str] = None, **params) -> int:
        """Create a Modelfile from a Hugging Face model.

        Args:
            model_id: The Hugging Face model identifier (or URL).
            output: Output file path (optional).
            **params: Additional parameters (context_length, temperature, top_p, num_threads).

        Returns:
            Exit code (0 for success, 1 for error).
        """
        # Parse and normalize model ID (handles URLs)
        model_id = self._parse_model_id(model_id)

        # Fetch model info
        self.console.print("[bold cyan]Fetching model information...[/bold cyan]")
        model_info = self.hf_client.fetch_model(model_id)

        if model_info is None:
            self.console.print("[red]Error: Model not found on Hugging Face.[/red]")
            print_error(
                f"Model '{model_id}' not found. Try 'list' or 'search' to find available models."
            )
            return 1

        # Display model info
        info_text = get_model_info_display(model_info)
        self.console.print(info_text)

        # Generate Modelfile
        context_length = params.get("context_length", 4096)
        params_dict = {
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.9),
            "num_threads": params.get("num_threads", 4),
        }

        modelfile_content = self.modelfile_generator.generate(
            model_id, context_length, params_dict
        )

        # Save to file
        if output:
            success = self.modelfile_generator.save(modelfile_content, output)
            if not success:
                print_error("Failed to save Modelfile")
                return 1

            self.console.print(
                "[bold green]Modelfile generated successfully![/bold green]"
            )
            self.console.print(f"   Saved to: {output}")
        else:
            # Display in terminal
            self.console.print("\n[bold]Modelfile content:[/bold]")
            self.console.print("-" * 60)
            self.console.print(modelfile_content)
            self.console.print("-" * 60)

            # Ask to save
            self.console.print("\nSave to file? [y/N] ", end="")
            try:
                response = input().strip().lower()
                if response == "y":
                    output = input("Output file path: ")
                    self.modelfile_generator.save(modelfile_content, output)
                    self.console.print(
                        "[bold green]Saved to: {}/[/bold green]".format(output)
                    )
            except EOFError:
                pass

            return 0

        return 0

    def deploy(self, model_id: str, output: Optional[str] = None, **params) -> int:
        """Deploy a model to Ollama.

        This command:
        1. Fetches model information from Hugging Face
        2. Generates a logical model name
        3. Creates a Modelfile with default parameters
        4. Adds the model to Ollama

        Args:
            model_id: The Hugging Face model identifier (or URL).
            output: Optional output file path.
            **params: Additional parameters (context_length, temperature, top_p, num_threads).

        Returns:
            Exit code (0 for success, 1 for error).
        """
        # Parse and normalize model ID (handles URLs)
        model_id = self._parse_model_id(model_id)

        # Fetch model info
        self.console.print("[bold cyan]Fetching model information...[/bold cyan]")
        model_info = self.hf_client.fetch_model(model_id)

        if model_info is None:
            self.console.print("[red]Error: Model not found on Hugging Face.[/red]")
            print_error(
                f"Model '{model_id}' not found. Try 'list' or 'search' to find available models."
            )
            return 1

        # Generate a logical model name
        model_name = self._generate_model_name(model_id, model_info)
        self.console.print(f"[bold green]Model Name:[/bold green] {model_name}")

        # Display model info
        info_text = get_model_info_display(model_info)
        self.console.print(info_text)

        # Generate Modelfile with default parameters
        context_length = params.get("context_length", 4096)
        params_dict = {
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.9),
            "num_threads": params.get("num_threads", 4),
        }

        modelfile_content = self.modelfile_generator.generate(
            model_id, context_length, params_dict
        )

        # Add to Ollama
        self.console.print("\n[bold cyan]Adding model to Ollama...[/bold cyan]")
        try:
            import subprocess

            # Check if Ollama is running
            self._check_ollama()

            # Create the model
            result = subprocess.run(
                ["ollama", "create", model_name, "-f", "-", "<modelfile>"],
                input=modelfile_content,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                self.console.print("[red]Error creating model in Ollama:[/red]")
                self.console.print(result.stderr)
                return 1

            # Pull the model (required for ollama create)
            self.console.print("\n[bold cyan]Pulling model...[/bold cyan]")
            pull_result = subprocess.run(
                ["ollama", "pull", model_name], capture_output=True, text=True
            )

            if pull_result.returncode != 0:
                self.console.print("[red]Error pulling model:[/red]")
                self.console.print(pull_result.stderr)
                return 1

            # Show success message
            self.console.print(
                "\n[bold green]✓ Model deployed successfully![/bold green]"
            )
            self.console.print(f"   Model name: {model_name}")
            self.console.print(f"   Context length: {context_length} tokens")
            self.console.print("\n[bold]Run the model with:[/bold]")
            self.console.print(f"   ollama run {model_name}")

            # Optionally save modelfile to file
            if output:
                success = self.modelfile_generator.save(modelfile_content, output)
                if success:
                    self.console.print(
                        f"\n[bold green]Modelfile saved to: {output}[/bold green]"
                    )

            return 0

        except FileNotFoundError:
            self.console.print(
                "[red]Error: Ollama is not installed or not in PATH.[/red]"
            )
            self.console.print("\nInstall Ollama from: https://ollama.com")
            return 1
        except subprocess.SubprocessError as e:
            self.console.print("[red]Error deploying model:[/red] {}/".format(e))
            return 1

    def _check_ollama(self) -> None:
        """Check if Ollama is running."""
        import subprocess

        try:
            result = subprocess.run(
                ["ollama", "ps"], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0 or not result.stdout.strip():
                self.console.print(
                    "[yellow]Warning: No models found. Starting Ollama...[/yellow]"
                )
                result = subprocess.run(
                    ["ollama", "serve"], capture_output=True, text=True
                )
                # Don't wait for ollama serve to complete
        except FileNotFoundError:
            pass  # Ollama not installed
        except subprocess.TimeoutExpired:
            self.console.print(
                "[yellow]Warning: Ollama may not be responding.[/yellow]"
            )

    def _generate_model_name(self, model_id: str, model_info: Any) -> str:
        """Generate a logical model name from the model ID.

        Args:
            model_id: The Hugging Face model ID.
            model_info: Model information object.

        Returns:
            A logical model name suitable for Ollama.
        """
        # Extract base name from model ID
        # Handle formats like: "jackrong/qwen3.5", "hf-coherence/command-r-plus"
        parts = model_id.split("/")

        if len(parts) < 2:
            return model_id

        # Get the model name (second part)
        base_name = parts[1]

        # Clean up the name for Ollama (replace special characters)
        import re

        clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", base_name)[:25]

        # Add version/quality suffix if available from tags
        version_suffix = ""
        tags = getattr(model_info, "tags", [])

        # Look for quantization info in tags
        for tag in tags:
            if "q3" in tag.lower() or "q3k" in tag.lower():
                version_suffix = "-q3k_m"
                break
            elif "q4" in tag.lower():
                version_suffix = "-q4_0"
                break
            elif "q5" in tag.lower():
                version_suffix = "-q5_0"
                break
            elif "q6" in tag.lower():
                version_suffix = "-q6_k"
                break
            elif "q8" in tag.lower():
                version_suffix = "-q8_0"
                break

        # Add context length suffix if available
        context_suffix = ""
        try:
            # Look for context length in tags
            for tag in tags:
                if "context_length" in tag.lower():
                    ctx_len = tag.split("=")[1].split(",")[0]
                    context_suffix = f"-{ctx_len}"
                    break
        except Exception:
            pass

        # Combine all parts
        model_name = f"{clean_name}{version_suffix}{context_suffix}"

        self.console.print(f"[bold yellow]Generated name:[/bold yellow] {model_name}")
        return model_name

    def convert(self, model_id: str, output: Optional[str] = None, **params) -> int:
        """Convert a Hugging Face model to Ollama format.

        Args:
            model_id: The Hugging Face model identifier (or URL).
            output: Output file path (optional).
            **params: Additional parameters (context_length, temperature, top_p, num_threads).

        Returns:
            Exit code (0 for success, 1 for error).
        """
        # Parse and normalize model ID (handles URLs)
        model_id = self._parse_model_id(model_id)

        # Fetch model info
        self.console.print("[bold cyan]Fetching model information...[/bold cyan]")
        model_info = self.hf_client.fetch_model(model_id)

        if model_info is None:
            self.console.print("[red]Error: Model not found on Hugging Face.[/red]")
            print_error(
                f"Model '{model_id}' not found. Try 'list' or 'search' to find available models."
            )
            return 1

        # Display model info
        info_text = get_model_info_display(model_info)
        self.console.print(info_text)

        # Convert to Ollama format
        context_length = params.get("context_length", 4096)
        params_dict = {
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.9),
            "num_threads": params.get("num_threads", 4),
        }

        ollama_def = self.ollama_converter.convert_to_ollama(
            model_id, context_length, params_dict
        )

        # Save to file
        if output:
            success = self.modelfile_generator.save(ollama_def, output)
            if not success:
                print_error("Failed to save Ollama model definition")
                return 1

            self.console.print(
                "[bold green]Ollama model definition generated successfully![/bold green]"
            )
            self.console.print(f"   Saved to: {output}")
        else:
            # Display in terminal
            self.console.print("\n[bold]Ollama model definition:[/bold]")
            self.console.print("-" * 60)
            self.console.print(ollama_def)
            self.console.print("-" * 60)

            # Ask to save
            self.console.print("\nSave to file? [y/N] ", end="")
            try:
                response = input().strip().lower()
                if response == "y":
                    output = input("Output file path: ")
                    self.modelfile_generator.save(ollama_def, output)
                    self.console.print(
                        "[bold green]Saved to: {}/[/bold green]".format(output)
                    )
            except EOFError:
                pass

            return 0

        return 0

    def list_models(self, category: Optional[str] = None) -> int:
        """List available Hugging Face models.

        Args:
            category: Category to filter by.

        Returns:
            Exit code (0 for success, 1 for error).
        """
        self.console.print("[bold cyan]Listing models...[/bold cyan]")

        try:
            models = self.hf_client.list_models(category=category)
        except Exception as e:
            self.console.print("[red]Error:[/red] Failed to list models: {}/".format(e))
            return 1

        # Create table
        table = Table(title="Models")
        table.add_column("", style="dim")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Tags", style="blue")
        table.add_column("Files", style="dim")

        for model in models[:100]:  # Limit to first 100
            tags = ", ".join(model.tags[:3])
            if len(model.tags) > 3:
                tags += f" (+{len(model.tags) - 3} more)"

            table.add_row(
                "",
                model.id,
                model.name,
                tags,
                str(len(model.files_metadata)) if model.files_metadata else "0",
            )

        # Add footer
        table.add_row(
            "",
            "",
            "",
            "",
            "",
            style="dim",
        )
        table.add_row(
            "",
            "",
            "",
            "",
            f"{len(models)} total models",
            style="dim",
        )

        self.console.print(table)

        if len(models) > 100:
            self.console.print(
                "\n[italic]Showing first 100 of {}/ models. Use 'search' to find specific models.[/italic]".format(
                    len(models)
                )
            )

        return 0

    def search_models(self, query: str) -> int:
        """Search for Hugging Face models.

        Args:
            query: Search query string.

        Returns:
            Exit code (0 for success, 1 for error).
        """
        self.console.print(
            "[bold cyan]Searching for '{}...'[/bold cyan]".format(query[:30])
        )

        try:
            models = self.hf_client.search_models(query)
        except Exception as e:
            self.console.print(
                "[red]Error:[/red] Failed to search models: {}/".format(e)
            )
            return 1

        # Create table
        table = Table(title="Search Results")
        table.add_column("", style="dim")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Tags", style="blue")
        table.add_column("Files", style="dim")

        for model in models[:50]:  # Limit to first 50
            tags = ", ".join(model.tags[:3])
            if len(model.tags) > 3:
                tags += f" (+{len(model.tags) - 3} more)"

            table.add_row(
                "",
                model.id,
                model.name,
                tags,
                str(len(model.files_metadata)) if model.files_metadata else "0",
            )

        # Add footer
        table.add_row(
            "",
            "",
            "",
            "",
            "",
            style="dim",
        )
        table.add_row(
            "",
            "",
            "",
            "",
            f"{len(models)} total results",
            style="dim",
        )

        self.console.print(table)

        if len(models) > 50:
            self.console.print(
                "\n[italic]Showing first 50 of {}/ results. Try a different search term.[/italic]".format(
                    len(models)
                )
            )

        return 0

    def list_templates(self) -> int:
        """List available templates.

        Returns:
            Exit code (0 for success, 1 for error).
        """
        self.console.print("[bold cyan]Available templates:[/bold cyan]")

        templates = self.template_manager.get_templates()

        if not templates:
            self.console.print("[yellow]No templates found.[/yellow]")
            return 0

        table = Table(title="Templates")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("File", style="dim")

        for template in templates:
            description = ""
            try:
                with open(f"templates/{template}.j2", "r", encoding="utf-8") as f:
                    description = f.read().split("{%")
                    if len(description) > 1:
                        description = description[1].split("%}")[0].strip()[:50]
            except Exception:
                pass

            table.add_row(
                template,
                description or "",
                f"templates/jinja2/{template}.j2",
            )

        self.console.print(table)

        return 0

    def interactive(self) -> int:
        """Run interactive wizard mode.

        Returns:
            Exit code (0 for success, 1 for error).
        """
        self.console.print("\n[bold]Interactive Modelfile Wizard[/bold]\n")

        try:
            # Get model selection
            self.console.print("1. Generate from Hugging Face model")
            self.console.print("2. List available models")
            self.console.print("3. Search models")
            choice = input("Choose option (1-3): ").strip()

            if choice == "1":
                model_id = input(
                    "Enter model ID (e.g., hf-coherence/command-r-plus): "
                ).strip()
                if not model_id:
                    print_error("Model ID is required")
                    return 1

                # Fetch and display model info
                model_info = self.hf_client.fetch_model(model_id)
                if model_info is None:
                    print_error(f"Model '{model_id}' not found")
                    return 1

                info_text = get_model_info_display(model_info)
                self.console.print(info_text)

                # Generate Modelfile
                context_length = int(input("Context length (tokens): ") or "4096")

                params = {
                    "temperature": float(input("Temperature (0-1): ") or "0.7"),
                    "top_p": float(input("Top-p (0-1): ") or "0.9"),
                    "num_threads": int(input("Number of threads: ") or "4"),
                }

                modelfile_content = self.modelfile_generator.generate(
                    model_id, context_length, params
                )

                # Save
                output = input("Save to file? (y/N): ").strip().lower()
                if output == "y":
                    filepath = input("File path: ")
                    self.modelfile_generator.save(modelfile_content, filepath)
                    self.console.print(
                        "[bold green]Saved to: {}/[/bold green]".format(filepath)
                    )
                else:
                    self.console.print("\n[bold]Modelfile content:[/bold]")
                    self.console.print(modelfile_content)

                return 0

            elif choice == "2":
                return self.list_models()

            elif choice == "3":
                query = input("Search query: ").strip()
                if query:
                    return self.search_models(query)
                else:
                    print_error("Search query cannot be empty")
                    return 1

            else:
                print_error("Invalid choice")
                return 1

        except KeyboardInterrupt:
            self.console.print("\n\n[red]Cancelled by user.[/red]")
            return 0
        except EOFError:
            self.console.print("\n\n[red]Input ended unexpectedly.[/red]")
            return 0

        return 1

    def show_help(self) -> int:
        """Display help information.

        Returns:
            Exit code (0 for success).
        """
        help_text = """
╭────────────────────────────────────────────────────────────────────────────────╮
│                    cookie-cutter-modelfile - Modelfile Generator                 │
│                    Generate Ollama Modelfiles from Hugging Face models           │
╰────────────────────────────────────────────────────────────────────────────────╯

COMMANDS:

  create [model_id] [OPTIONS]
      Generate a Modelfile from a Hugging Face model

  convert [model_id] [OPTIONS]
      Convert a model to Ollama format

  deploy [model_id] [OPTIONS]
      Generate Modelfile and add model to Ollama automatically

  list [CATEGORY]
      List available Hugging Face models

  search [QUERY]
      Search for models by query

  template
      List available templates

  interactive
      Run interactive wizard mode

  --help
      Show this help message


OPTIONS:

  --output, -o FILE
      Output file path for Modelfile

  --context-length, -c INTEGER
      Context window size (default: 4096)

  --temperature FLOAT
      Temperature parameter (default: 0.7)

  --top-p FLOAT
      Top-p parameter (default: 0.9)

  --num-threads INTEGER
      Number of threads (default: 4)


EXAMPLES:

  # Deploy a model from URL (recommended!)
  $ cookie-cutter-modelfile deploy "https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/resolve/main/Qwen3.5-9B.Q3_K_M.gguf?download=true"

  # Deploy a model by ID
  $ cookie-cutter-modelfile deploy jackrong/qwen3.5

  # Deploy with custom context length
  $ cookie-cutter-modelfile deploy "https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF" --context-length 8192

  # Create Modelfile without deploying
  $ cookie-cutter-modelfile create "https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF" --output my-model.modelfile

  # Convert to Ollama format
  $ cookie-cutter-modelfile convert "https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF" --output model.ollama

  # Interactive mode
  $ cookie-cutter-modelfile interactive


AUTO-GENERATED MODEL NAMES:

  The deploy command automatically generates logical names:
  - jackrong/qwen3.5 → qwen3.5-q3k_m
  - hf-coherence/command-r-plus → command-r-plus
  - openbmb/minicpm5-2b → minicpm5-2b-f16

  Names include:
  - Base model name (cleaned)
  - Quantization level (q3k_m, q4_0, q5_0, q6_k, q8_0)
  - Context length (when available from tags)

  ╭────────────────────────────────────────────────────────────────────────────────╮
│                    Use 'cookie-cutter-modelfile --help' for more details         │
╰────────────────────────────────────────────────────────────────────────────────╯
        """
        self.console.print(help_text)
        return 0

    def run(self, args: List[str]) -> int:
        """Run the CLI with the given arguments.

        Args:
            args: Command line arguments.

        Returns:
            Exit code.
        """
        parser = argparse.ArgumentParser(
            prog="cookie-cutter-modelfile",
            description="Generate Ollama Modelfiles from Hugging Face models",
            add_help=False,
        )

        subparsers = parser.add_subparsers(dest="command", help="Command")

        # Create command
        create_parser = subparsers.add_parser(
            "create",
            help="Generate Modelfile from Hugging Face model",
        )
        create_parser.add_argument(
            "model_id",
            help="Hugging Face model identifier",
        )
        create_parser.add_argument(
            "--output",
            "-o",
            help="Output file path",
        )
        create_parser.add_argument(
            "--template",
            "-t",
            help="Template to use",
            choices=["chat", "completion", "image", "default"],
        )
        create_parser.add_argument(
            "--context-length",
            "-c",
            type=int,
            help="Context window size",
        )
        create_parser.add_argument(
            "--temperature",
            "-T",
            type=float,
            help="Sampling temperature",
        )
        create_parser.add_argument(
            "--top-p",
            "-P",
            type=float,
            help="Top-p sampling",
        )
        create_parser.add_argument(
            "--num-threads",
            "-n",
            type=int,
            help="Number of threads",
        )
        create_parser.add_argument(
            "--num-images",
            "-i",
            type=int,
            help="Number of images",
        )
        create_parser.add_argument(
            "--size",
            "-s",
            help="Image size",
        )

        # Convert command
        convert_parser = subparsers.add_parser(
            "convert",
            help="Convert to Ollama format",
        )
        convert_parser.add_argument(
            "model_id",
            help="Hugging Face model identifier",
        )
        convert_parser.add_argument(
            "--output",
            "-o",
            help="Output file path",
        )
        convert_parser.add_argument(
            "--context-length",
            "-c",
            type=int,
            help="Context window size",
        )
        convert_parser.add_argument(
            "--temperature",
            "-T",
            type=float,
            help="Sampling temperature",
        )
        convert_parser.add_argument(
            "--top-p",
            "-P",
            type=float,
            help="Top-p sampling",
        )
        convert_parser.add_argument(
            "--num-threads",
            "-n",
            type=int,
            help="Number of threads",
        )

        # Deploy command
        deploy_parser = subparsers.add_parser(
            "deploy",
            help="Generate Modelfile and add model to Ollama automatically",
        )
        deploy_parser.add_argument(
            "model_id",
            help="Hugging Face model identifier (or URL)",
        )
        deploy_parser.add_argument(
            "--output",
            "-o",
            help="Output file path for Modelfile",
        )
        deploy_parser.add_argument(
            "--context-length",
            "-c",
            type=int,
            help="Context window size",
        )
        deploy_parser.add_argument(
            "--temperature",
            "-T",
            type=float,
            help="Sampling temperature",
        )
        deploy_parser.add_argument(
            "--top-p",
            "-P",
            type=float,
            help="Top-p sampling",
        )
        deploy_parser.add_argument(
            "--num-threads",
            "-n",
            type=int,
            help="Number of threads",
        )

        # List command
        list_parser = subparsers.add_parser(
            "list",
            dest="command",
            help="List available models",
        )
        list_parser.add_argument(
            "category",
            help="Category to filter by",
        )

        # Search command
        search_parser = subparsers.add_parser(
            "search",
            dest="command",
            help="Search for models",
        )
        search_parser.add_argument(
            "query",
            help="Search query",
        )

        # Template command
        template_parser = subparsers.add_parser(
            "template",
            dest="command",
            help="List available templates",
        )

        # Interactive command
        interactive_parser = subparsers.add_parser(
            "interactive",
            dest="command",
            help="Interactive wizard mode",
        )

        # Help command
        help_parser = subparsers.add_parser(
            "--help",
            "-h",
            dest="command",
            help="Show help message",
        )

        args = parser.parse_args(args)

        # Dispatch to command
        if args.command == "create":
            return self.create(
                args.model_id,
                args.output,
                **{
                    "context_length": args.context_length,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "num_threads": args.num_threads,
                    "num_images": args.num_images,
                    "size": args.size,
                },
            )

        elif args.command == "convert":
            return self.convert(
                args.model_id,
                args.output,
                **{
                    "context_length": args.context_length,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "num_threads": args.num_threads,
                },
            )

        elif args.command == "list":
            return self.list_models(args.category)

        elif args.command == "search":
            return self.search_models(args.query)

        elif args.command == "template":
            return self.list_templates()

        elif args.command == "interactive":
            return self.interactive()

        elif args.command == "--help" or args.command is None:
            return self.show_help()

        else:
            parser.print_help()
            return 1
        create_parser.add_argument(
            "--output",
            "-o",
            help="Output file path",
        )
        create_parser.add_argument(
            "--template",
            "-t",
            help="Template to use",
            choices=["chat", "completion", "image", "default"],
        )
        create_parser.add_argument(
            "--context-length",
            "-c",
            type=int,
            help="Context window size",
        )
        create_parser.add_argument(
            "--temperature",
            "-T",
            type=float,
            help="Sampling temperature",
        )
        create_parser.add_argument(
            "--top-p",
            "-P",
            type=float,
            help="Top-p sampling",
        )
        create_parser.add_argument(
            "--num-threads",
            "-n",
            type=int,
            help="Number of threads",
        )
        create_parser.add_argument(
            "--num-images",
            "-i",
            type=int,
            help="Number of images",
        )
        create_parser.add_argument(
            "--size",
            "-s",
            help="Image size",
        )

        # Convert command
        convert_parser = subparsers.add_parser(
            "convert",
            help="Convert to Ollama format",
        )
        convert_parser.add_argument(
            "model_id",
            help="Hugging Face model identifier",
        )
        convert_parser.add_argument(
            "--output",
            "-o",
            help="Output file path",
        )
        convert_parser.add_argument(
            "--context-length",
            "-c",
            type=int,
            help="Context window size",
        )
        convert_parser.add_argument(
            "--temperature",
            "-T",
            type=float,
            help="Sampling temperature",
        )
        convert_parser.add_argument(
            "--top-p",
            "-P",
            type=float,
            help="Top-p sampling",
        )
        convert_parser.add_argument(
            "--num-threads",
            "-n",
            type=int,
            help="Number of threads",
        )

        # Deploy command
        deploy_parser = subparsers.add_parser(
            "deploy",
            help="Generate Modelfile and add model to Ollama automatically",
        )
        deploy_parser.add_argument(
            "model_id",
            help="Hugging Face model identifier (or URL)",
        )
        deploy_parser.add_argument(
            "--output",
            "-o",
            help="Output file path for Modelfile",
        )
        deploy_parser.add_argument(
            "--context-length",
            "-c",
            type=int,
            help="Context window size",
        )
        deploy_parser.add_argument(
            "--temperature",
            "-T",
            type=float,
            help="Sampling temperature",
        )
        deploy_parser.add_argument(
            "--top-p",
            "-P",
            type=float,
            help="Top-p sampling",
        )
        deploy_parser.add_argument(
            "--num-threads",
            "-n",
            type=int,
            help="Number of threads",
        )

        # List command
        list_parser = subparsers.add_parser(
            "list",
            dest="command",
            help="List available models",
        )
        list_parser.add_argument(
            "category",
            help="Category to filter by",
        )

        # Search command
        search_parser = subparsers.add_parser(
            "search",
            dest="command",
            help="Search for models",
        )
        search_parser.add_argument(
            "query",
            help="Search query",
        )

        # Template command
        template_parser = subparsers.add_parser(
            "template",
            dest="command",
            help="List available templates",
        )

        # Interactive command
        interactive_parser = subparsers.add_parser(
            "interactive",
            dest="command",
            help="Interactive wizard mode",
        )

        # Help command
        help_parser = subparsers.add_parser(
            "--help",
            "-h",
            dest="command",
            help="Show help message",
        )

        args = parser.parse_args(args)

        # Dispatch to command
        if args.command == "create":
            return self.create(
                args.model_id,
                args.output,
                **{
                    "context_length": args.context_length,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "num_threads": args.num_threads,
                    "num_images": args.num_images,
                    "size": args.size,
                },
            )

        elif args.command == "convert":
            return self.convert(
                args.model_id,
                args.output,
                **{
                    "context_length": args.context_length,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "num_threads": args.num_threads,
                },
            )

        elif args.command == "list":
            return self.list_models(args.category)

        elif args.command == "search":
            return self.search_models(args.query)

        elif args.command == "template":
            return self.list_templates()

        elif args.command == "interactive":
            return self.interactive()

        elif args.command == "--help" or args.command is None:
            return self.show_help()

        else:
            parser.print_help()
            return 1


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code.
    """
    cli = CLI()

    args = cli.parse_args()

    if args.command == "create":
        return cli.create(
            args.model_id,
            args.output,
            **{
                "context_length": args.context_length,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_threads": args.num_threads,
                "num_images": args.num_images,
                "size": args.size,
            },
        )
    elif args.command == "convert":
        return cli.convert(
            args.model_id,
            args.output,
            **{
                "context_length": args.context_length,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_threads": args.num_threads,
            },
        )
    elif args.command == "deploy":
        return cli.deploy(
            args.model_id,
            args.output,
            **{
                "context_length": args.context_length,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_threads": args.num_threads,
            },
        )
    elif args.command == "list":
        return cli.list_models(args.category)
    elif args.command == "search":
        return cli.search_models(args.query)
    elif args.command == "template":
        return cli.list_templates()
    elif args.command == "interactive":
        return cli.interactive()
    else:
        return cli.show_help()


if __name__ == "__main__":
    sys.exit(main())
