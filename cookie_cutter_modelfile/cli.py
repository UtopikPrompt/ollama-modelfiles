"""
Command-line interface for the cookie cutter modelfile tool.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .config import load_config, print_config
from .context import VRAM_RESERVE, estimate_context, format_context
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

# Folder where deploy() writes the temporary Modelfile before running
# `ollama create`. It lives in the project (not the system temp) so it can be
# tracked or excluded consistently, and is added to .gitignore.
OLLAMA_CREATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "modelfiles"
)


class CLI:
    """Command-line interface for the modelfile tool."""

    def __init__(self):
        """Initialize the CLI."""
        self.console = Console(file=sys.stdout)
        self.hf_client = HFClient()
        self.template_manager = TemplateManager()
        self.config = load_config()
        self.config_loaded = True
        self.modelfile_generator = ModelfileGenerator()
        self.ollama_converter = OllamaConverter()

    def print_config(self):
        """Print the resolved Ollama URL and available hardware profiles."""
        print_config(self.config)

    def _apply_hardware(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Inject num_thread / num_gpu from the active hardware profile.

        Args:
            params: The params dict to annotate (num_thread key already present).

        Returns:
            The same dict, with ``num_thread`` and ``num_gpu`` set from
            the resolved hardware profile (CLI/env overrides still apply).
        """
        hardware = self.config.get_hardware(params.get("hardware"))
        params["num_thread"] = self.config.get_num_threads(params.get("hardware"))
        params["num_gpu"] = self.config.get_gpu_layers(params.get("hardware"))
        return params

    def _ollama_binary(self, override: Optional[str] = None) -> str:
        """Return the Ollama binary to invoke.

        Returns ``override`` when given, otherwise the configured URL. When the
        URL differs from the default ``http://localhost:11434``, we point the
        ``ollama`` CLI at it via ``ollama --host <url>``; otherwise we just use
        the bare ``ollama`` binary (equivalent to ``--host http://localhost:11434``).

        Args:
            override: Optional explicit URL override.

        Returns:
            Either ``"ollama"`` or ``["ollama", "--host", <url>"]``.
        """
        url = self.config.get_ollama_url(override)
        if override is not None:
            return override
        if url == "http://localhost:11434":
            return "ollama"
        return ["ollama", "--host", url]

    def parse_args(self) -> argparse.Namespace:
        """Parse command line arguments.

        Returns:
            Parsed arguments namespace.
        """
        # Create a separate top-level parser for --help/-h
        top_level_parser = argparse.ArgumentParser(
            prog="cookie-cutter-modelfile",
            add_help=False,
        )
        top_level_parser.add_argument(
            "-h", "--help", action="help", help="show help message and exit"
        )
        top_level_parser.add_argument("command", nargs="*", help="Command to run")

        # Parse with top-level parser to detect --help/-h
        try:
            top_level_args = top_level_parser.parse_args(args=sys.argv[1:])
            # If --help/-h was used, argparse will exit, so we catch it
            if not top_level_args.command:
                top_level_parser.exit()
        except SystemExit:
            # --help/-h was used, argparse exits cleanly
            return None
        self.top_level_args = top_level_args

        # If --help/-h wasn't used, create the subparsers parser
        self.parser = argparse.ArgumentParser(
            prog="cookie-cutter-modelfile",
            description="Generate Ollama Modelfiles from Hugging Face models",
        )
        self.subparsers = self.parser.add_subparsers(help="Command")

        # Create command parsers
        create_parser = self.subparsers.add_parser(
            "create",
            help="Create a Modelfile from a Hugging Face model",
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
        create_parser.add_argument(
            "--hardware",
            help="Hardware profile name (see `--help` or the docs)",
        )

        # Convert command
        convert_parser = self.subparsers.add_parser(
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
        deploy_parser = self.subparsers.add_parser(
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
        deploy_parser.add_argument(
            "--hardware",
            help="Hardware profile name (see `--help` or the docs)",
        )

        # List command
        list_parser = self.subparsers.add_parser(
            "list",
            help="List available models",
        )
        list_parser.add_argument(
            "category",
            help="Category to filter by",
        )

        # Search command
        search_parser = self.subparsers.add_parser(
            "search",
            help="Search for models",
        )
        search_parser.add_argument(
            "query",
            help="Search query",
        )

        # Template command
        template_parser = self.subparsers.add_parser(
            "template",
            help="Show available templates",
        )

        # Interactive command
        interactive_parser = self.subparsers.add_parser(
            "interactive",
            help="Interactive wizard mode",
        )

        # Re-parse with subparsers so the chosen command and its model_id are populated
        parsed = self.parser.parse_args(args=self.top_level_args.command)
        parsed.command = self.top_level_args.command[0]
        return parsed

    def _strip_none(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of params with all None values removed.

        Args:
            params: Dictionary that may contain None values.

        Returns:
            New dict containing only entries whose value is not None.
        """
        return {k: v for k, v in params.items() if v is not None}

    def run(self, args: argparse.Namespace) -> int:
        """Run the CLI with the given arguments.

        Args:
            args: Command line arguments.

        Returns:
            Exit code.
        """
        if args.command == "create":
            return self.create(
                args.model_id,
                args.output,
                **self._strip_none(
                    {
                        "context_length": args.context_length,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "num_thread": args.num_threads,
                        "num_images": args.num_images,
                        "size": args.size,
                        "hardware": args.hardware,
                    }
                ),
            )
        elif args.command == "convert":
            return self.convert(
                args.model_id,
                args.output,
                **self._strip_none(
                    {
                        "context_length": args.context_length,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "num_thread": args.num_threads,
                        "hardware": args.hardware,
                    }
                ),
            )
        elif args.command == "deploy":
            return self.deploy(
                args.model_id,
                args.output,
                **self._strip_none(
                    {
                        "context_length": args.context_length,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "num_thread": args.num_threads,
                        "hardware": args.hardware,
                    }
                ),
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
            self.parser.print_help()
            return 1

    def _extract_quant_from_url(self, url: str) -> Optional[str]:
        """Extract the quantization from a raw Hugging Face URL.

        Parses the ``/resolve/main/FNAME.gguf`` (or ``/tree/...`` / ``/blob/...``)
        segment, e.g. ``Ornith-1.5-9B-uncensored.Q3_K_M.gguf`` -> ``Q3_K_M``.
        This must be done before ``_parse_model_id`` normalizes the URL, which
        discards the GGUF filename and its quant tag.

        The quantization tag keeps its **original case** because Hugging Face
        URLs are case-sensitive: ``Q3_K_L.gguf`` is a different file than
        ``q3_k_l.gguf`` and the latter 404s. Only the quant tag (e.g.
        ``Q3_K_L``) is returned — not the full filename — because it is later
        embedded into ``{base}.{quant}.gguf``.

        Args:
            url: The raw Hugging Face URL (may contain a /resolve/main/ path).

        Returns:
            The quantization tag (e.g. ``Q3_K_M``) preserving case, or None.
        """
        if not url:
            return None
        # Match the GGUF filename in the /resolve/<ref>/<filename> path (also
        # covers /tree/<ref>/ and /blob/<ref>/ browser URLs). We preserve the
        # original case of the filename's quant tag for Hugging Face.
        import re

        match = re.search(r"/(?:resolve|tree|blob)/[^/]+/([^/]+)\.gguf", url)
        if match:
            filename = match.group(1)  # e.g. "Ornith-1.5-9B-uncensored.Q3_K_L"
            # A quant tag is a distinct component of the filename: the part
            # after the final dot. "ornith.gguf" has no quant (stem has no
            # dot) -> return None so the deployer picks a quant normally.
            # "ornith.Q3_K_L.gguf" -> quant "Q3_K_L". Return only that tag,
            # preserving its original case (HF is case-sensitive).
            if "." in filename:
                return filename.rsplit(".", 1)[-1]
            return None
        return None

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
            "num_thread": params.get("num_thread", 4),
        }
        params_dict = self._apply_hardware(params_dict)

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

    def _requested_gguf_filename(
        self, model_id: str, requested_quant: Optional[str], best: Optional[str] = None
    ) -> str:
        """Build the GGUF filename to download based on the requested quant.

        Priority: the exact ``requested_quant`` from the URL, then VRAM-best
        (``best``), then the base filename.

        Args:
            model_id: The normalized Hugging Face model identifier.
            requested_quant: Optional quantization tag requested in the URL
                (e.g. ``q3_k_m``).
            best: Optional VRAM-derived quantization (legacy fallback).

        Returns:
            The GGUF filename (e.g. ``Ornith-1.5-9B-uncensored.Q3_K_M.gguf``).
        """
        # When a quant is explicitly requested (e.g. from the URL), it is
        # authoritative: build the filename directly from the repo base name
        # WITHOUT listing the repo files (which would return a different/worse
        # quant first). Fall back to the full derivation only when no quant is
        # requested.
        if requested_quant:
            base = self._repo_base_name(model_id, best)
            return f"{base}.{requested_quant}.gguf"
        return self._derive_gguf_filename(model_id, best)

    def _available_quants(self, model_id: str) -> list:
        """List quantizations available in the repo, sorted by preference.

        Args:
            model_id: The normalized Hugging Face model identifier.

        Returns:
            A list of quantization tags (e.g. ``[q4_k_m, q3_k_m]``).
        """
        from huggingface_hub import list_repo_files

        try:
            files = list_repo_files(model_id, repo_type="model")
        except Exception:
            return []
        order = [
            "f16",
            "q8_0",
            "q6_k",
            "q5_k_m",
            "q5_k_s",
            "q4_k_m",
            "q4_k_s",
            "q3_k_l",
            "q3_k_m",
            "q3_k_s",
            "iq4_xs",
            "q2_k",
        ]
        quants = []
        for f in files:
            if not f.endswith(".gguf"):
                continue
            quant = self._extract_quant_from_filename(f)
            if quant and quant not in quants:
                quants.append(quant)
        return [q for q in order if q in quants]

    def _prompt_quant(
        self, available: list, requested_filename: str, filename: str
    ) -> Optional[str]:
        """Prompt the user for a quantization choice.

        Args:
            available: Quantizations available in the repo, sorted by preference.
            requested_filename: Filename derived from any URL hint (for display).
            filename: Base filename (for display when no hint was provided).

        Returns:
            The chosen quantization tag (e.g. ``q4_k_m``) or None.
        """
        if not available:
            # No quantizations discovered: offer a sensible default.
            self.console.print(
                "[yellow]No quantizations detected in the repo.[/yellow]"
            )
            self.console.print(
                "[dim]Defaulting to q4_k_m (adjust the FROM line as needed).[/dim]"
            )
            return "q4_k_m"
        self.console.print("[yellow]Which quantization would you like?[/yellow]")
        for i, q in enumerate(available, 1):
            self.console.print(f"[dim]  {i}. {q}[/dim]")
        self.console.print(f"[dim]  {len(available) + 1}. custom[/dim]")
        try:
            raw = self.console.input(
                f"[dim]Select a number (1-{len(available) + 1}):[/dim] "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw.lower() in ("q", "quit", "exit"):
            return None
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(available):
                return available[idx - 1]
        return self._custom_quant(available)

    def _custom_quant(self, available: list) -> Optional[str]:
        """Prompt the user for a custom quantization string.

        Args:
            available: Quantizations available in the repo, for validation.

        Returns:
            The chosen quantization tag (e.g. ``q4_k_m``) or None.
        """
        import re

        while True:
            try:
                raw = (
                    self.console.input(
                        "[yellow]Enter a quantization (e.g. q4_k_m):[/yellow] "
                    )
                    .strip()
                    .lower()
                )
            except (EOFError, KeyboardInterrupt):
                return None
            if raw in ("q", "quit", "exit", ""):
                return None
            if re.fullmatch(r"[a-z0-9_.]+", raw):
                if raw in available:
                    return raw
                self.console.print(
                    f"[yellow]Warning: '{raw}' is not in the available list. "
                    f"Proceeding anyway.[/yellow]"
                )
                return raw
            self.console.print("[yellow]Invalid quantization. Try again.[/yellow]")

    def _repo_base_name(self, model_id: str, best: Optional[str] = None) -> str:
        """Derive the repo base name from the model ID.

        The repo is named ``{base}-GGUF`` and holds files like
        ``{base}.Q3_K_M.gguf``. We strip the ``-GGUF`` marker and, when a
        quantization is provided (``best``), construct ``{base}.{best}.gguf``.

        This deliberately does NOT list the repo files: when a quantization is
        explicitly requested it must be authoritative, so we never guess a
        different quant based on what the repo contains.

        Args:
            model_id: The normalized Hugging Face model identifier.
            best: Optional quantization tag (e.g. ``q3_k_m``).

        Returns:
            The derived GGUF filename (e.g. ``Ornith-1.5-9B-uncensored.Q3_K_M.gguf``).
        """
        parts = model_id.split("/")
        base = parts[-1] if len(parts) >= 2 else model_id
        # Strip the repo convention marker "-GGUF" (added to the repo name).
        if base.endswith("-GGUF") or base.endswith("-gguf"):
            base = base[: -len("-GGUF")]

        if best:
            # Construct "{base}.{best}.gguf" (e.g. Ornith-1.5-9B-uncensored.Q3_K_M.gguf)
            return f"{base}.{best}.gguf"

        # No best quant: return the bare base name.
        return base

    def _derive_gguf_filename(self, model_id: str, best: Optional[str] = None) -> str:
        """Derive the GGUF filename from the model ID.

        The repo is named ``{base}-GGUF`` and holds files like
        ``{base}.Q3_K_M.gguf``. We strip the ``-GGUF`` marker, then, if no
        quantization has been pinned, list the repo files and return the first
        ``.gguf`` (so the user gets a valid file even without an explicit
        quant).

        Args:
            model_id: The normalized Hugging Face model identifier.
            best: Optional VRAM-derived quantization (legacy fallback).

        Returns:
            The derived GGUF filename (e.g. ``Ornith-1.5-9B-uncensored.Q3_K_M.gguf``).
        """
        base = self._repo_base_name(model_id, best)

        # No best quant: list repo files to find the first .gguf
        try:
            from huggingface_hub import list_repo_files

            files = list_repo_files(model_id, repo_type="model")
            for f in files:
                if f.endswith(".gguf"):
                    return f
        except Exception:
            pass

        # Fallback: use base + first quant from preference order
        return f"{base}.gguf"

    def _extract_quant_from_filename(self, filename: str) -> str:
        """Extract the quantization tag from a GGUF filename.

        E.g. ``Ornith-1.5-9B-uncensored.Q3_K_M.gguf`` -> ``q3_k_m``.

        Args:
            filename: The GGUF filename (path or URL).

        Returns:
            The lowercase quantization tag (e.g. ``q3_k_m``) or ``""``.
        """
        import re

        base = os.path.basename(filename)
        # Strip a trailing ".gguf".
        if base.endswith(".gguf"):
            base = base[: -len(".gguf")]
        # The quantization tag is separated from the model name by a dot and
        # may itself contain underscores (e.g. "Q3_K_M"), so split on "." from
        # the right and take the final segment.
        name = re.split(r"[.]", base)[-1]
        return name.lower()

    def _best_quant(
        self,
        tags: Optional[list],
        vram_bytes: Optional[int],
        model_size_bytes: Optional[int],
        best_hint: Optional[str] = None,
    ) -> Optional[str]:
        """Pick the best quantization that fits VRAM-reserve model size.

        The preferred quantization is resolved from ``best_hint`` (the quant
        read from the downloaded GGUF filename) when available, falling back
        to the model tags, and finally to a VRAM-based scan of the size table.

        Args:
            tags: Quantization tags from model info.
            vram_bytes: Total VRAM in bytes.
            model_size_bytes: Exact on-disk bytes of the model (used to verify
                the candidate fits within ``VRAM_RESERVE`` of VRAM).
            best_hint: Optional quantization tag read from the GGUF filename
                (e.g. ``q3_k_m``). When provided it seeds the candidate.

        Returns:
            A quantization tag (e.g. ``q4_k_m``) or None.
        """
        if not vram_bytes or vram_bytes <= 0:
            return None
        # Preferred quantizations in order of increasing quality/size.
        order = [
            "f16",
            "q8_0",
            "q6_k",
            "q5_k_m",
            "q5_k_s",
            "q4_k_m",
            "q4_k_s",
            "q3_k_l",
            "q3_k_m",
            "q3_k_s",
            "iq4_xs",
            "q2_k",
        ]
        size_guess = {
            "f16": 17.9,
            "q8_0": 9.5,
            "q6_k": 7.4,
            "q5_k_m": 6.5,
            "q5_k_s": 6.3,
            "q4_k_m": 5.6,
            "q4_k_s": 5.4,
            "q3_k_l": 4.9,
            "q3_k_m": 4.6,
            "q3_k_s": 4.3,
            "iq4_xs": 5.2,
            "q2_k": 3.8,
        }
        usable = vram_bytes * (1 - VRAM_RESERVE)

        # Candidate resolution: hint (from filename) > tags > VRAM scan.
        candidate = best_hint
        if not candidate:
            # Detect the quantization from the model tags (longest keys first).
            for q in order:
                for tag in tags or []:
                    if q in tag.lower():
                        candidate = q
                        break
        if candidate and candidate in order:
            guess = size_guess.get(candidate.lower())
            if not model_size_bytes or model_size_bytes * guess / 1e9 <= usable:
                return candidate
            # Hint doesn't fit VRAM: fall through to the VRAM-based scan.

        # Scan the size table from best to worst and return the first quant
        # that fits within VRAM-reserve.
        for q in order:
            guess = size_guess.get(q.lower())
            if guess and model_size_bytes and model_size_bytes * guess / 1e9 > usable:
                continue
            return q
        return None

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
        # Extract the quantization requested in the URL (e.g. ...Q3_K_M.gguf).
        # Must be done BEFORE _parse_model_id normalizes the URL, which discards
        # the GGUF filename and its quant tag.
        requested_quant = self._extract_quant_from_url(model_id)
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

        # Determine the quantization to download: explicit URL > VRAM-best > default.
        tags = getattr(model_info, "tags", None)
        vram_bytes = self.config.get_vram(params.get("hardware"))
        # Initial filename guess (best may be refined after the download).
        best = None
        filename = self._derive_gguf_filename(model_id, best)

        # Detect the quantization requested in the original URL (e.g.
        # Ornith-1.5-9B-uncensored.Q3_K_M.gguf -> q3_k_m). When a quant is
        # requested we download exactly that filename and do NOT re-pick the
        # quant based on VRAM: the user asked for a specific file.
        requested_filename = self._requested_gguf_filename(
            model_id, requested_quant, best
        )
        # Download the GGUF file locally so the Modelfile FROM line
        # references a local path (ollama requires a local file, not a HF repo id).
        self.console.print("[bold cyan]Downloading GGUF model file...[/bold cyan]")
        gguf_path = None
        for attempt in [requested_filename, filename]:
            gguf_path = self.hf_client.download_gguf(model_id, attempt)
            if gguf_path:
                # Extract the actual quant from the downloaded filename so VRAM
                # context estimation uses real data.
                best_hint = self._extract_quant_from_filename(attempt)
                self.console.print(
                    f"[bold green]Downloaded:[/bold green] {attempt} -> {gguf_path}"
                )
                break
        if not gguf_path:
            self.console.print("[red]Error: GGUF file not found on Hugging Face.[/red]")
            print_error(f"Could not download a GGUF file from '{model_id}'.")
            return 1

        # Exact model size from the downloaded file (for VRAM context calc).
        import os

        model_size_bytes = os.path.getsize(gguf_path)

        # Best quant: prefer the real quant from the downloaded filename, then
        # fall back to a VRAM-based scan of the size table (tags alone are
        # unreliable for GGUF-derived repos that omit quant tags).
        # No quant auto-selection: the requested filename (from the URL) is
        # authoritative. If the link did not specify a quant, ask the user.
        if not requested_quant:
            available = self._available_quants(model_id)
            if available:
                self.console.print(
                    "[bold yellow]Quant not specified in the link.[/bold yellow]"
                )
                self.console.print(f"[dim]Available quantizations: {available}[/dim]")
                requested_quant = self._prompt_quant(
                    available, requested_filename, filename
                )
            else:
                self.console.print(
                    "[bold yellow]Quant not specified in the link and none available.[/bold yellow]"
                )
                requested_quant = "q4_k_m"  # sensible default when the user asks

        # Filename to download: the exact requested file when a quant was given,
        # otherwise the base filename.
        filename = requested_filename if requested_quant else filename

        # Compute a path RELATIVE to the Modelfile directory so Ollama resolves
        # the FROM line relative to the Modelfile itself (ollama loads the model
        # from a path accessible from the Modelfile's location). The GGUF is
        # downloaded to ./models while the Modelfile is written to
        # <root>/modelfiles, so the relative path bridges those two directories.
        # Falls back to the absolute path if the relative computation fails.
        try:
            modelfile_from = os.path.relpath(
                gguf_path, start=OLLAMA_CREATE_DIR
            ).replace("\\", "/")
        except Exception:
            modelfile_from = gguf_path.replace("\\", "/")

        # Generate Modelfile with default parameters, using the local GGUF path
        context_length = params.get("context_length")
        if not context_length:
            # Compute from VRAM - model_size when VRAM is known, else default.
            ctx = estimate_context(vram_bytes, model_size_bytes, tags)
            if ctx is None:
                context_length = 4096
            else:
                context_length = ctx

        params_dict = {
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.9),
            "num_thread": params.get("num_thread", 4),
        }

        # Inject num_thread / num_gpu from the active hardware profile
        params_dict = self._apply_hardware(params_dict)

        # Report the chosen context length and VRAM-derived sizing.
        if vram_bytes and model_size_bytes:
            self.console.print(
                "[bold yellow]Context length:[/bold yellow] "
                f"{format_context(context_length)} tokens "
                f"(from {vram_bytes//1e9:.1f}GB VRAM - {model_size_bytes//1e9:.2f}GB model)"
            )

        modelfile_content = self.modelfile_generator.generate(
            modelfile_from, context_length, params_dict
        )

        # Add to Ollama
        self.console.print("\n[bold cyan]Adding model to Ollama...[/bold cyan]")
        try:
            import subprocess
            import tempfile

            # Check if Ollama is running
            self._check_ollama()

            # Write the Modelfile to a dedicated folder (ollama requires a file
            # path, not stdin: `ollama create --file -` returns "no Modelfile found").
            # The temp Modelfile is created in a project-local folder that is
            # gitignored so it never shows up in version control.
            import subprocess

            modelfile_path = None
            try:
                os.makedirs(OLLAMA_CREATE_DIR, exist_ok=True)
                fd, modelfile_path = tempfile.mkstemp(
                    dir=OLLAMA_CREATE_DIR, suffix=".modelfile"
                )
                with open(modelfile_path, "w", encoding="utf-8") as tmp:
                    tmp.write(modelfile_content)

                # Create the model from the local Modelfile.
                # When the Modelfile FROM line points to a local GGUF file,
                # `ollama create` loads the model directly into the library.
                # `ollama pull` is NOT used: it only fetches from a remote
                # registry, and our Modelfile already references a downloaded
                # GGUF file on disk.
                result = subprocess.run(
                    ["ollama", "create", model_name, "--file", modelfile_path],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )

                if result.returncode != 0:
                    self.console.print("[red]Error creating model in Ollama:[/red]")
                    self.console.print(result.stderr)
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
            finally:
                # Always clean up the temp Modelfile
                if modelfile_path and os.path.exists(modelfile_path):
                    try:
                        os.remove(modelfile_path)
                    except OSError:
                        pass

        except FileNotFoundError:
            self.console.print(
                "[red]Error: Ollama is not installed or not in PATH.[/red]"
            )
            self.console.print("\nInstall Ollama from: https://ollama.com")
            return 1
        except subprocess.SubprocessError as e:
            self.console.print("[red]Error deploying model:[/red] {}/".format(e))
            return 1
        finally:
            # Always clean up the temp Modelfile
            if modelfile_path and os.path.exists(modelfile_path):
                try:
                    os.remove(modelfile_path)
                except OSError:
                    pass

    def _check_ollama(self) -> None:
        """Check if Ollama is running."""
        import subprocess

        try:
            result = subprocess.run(
                ["ollama", "ps"],
                capture_output=True,
                text=True,
                timeout=5,
                encoding="utf-8",
            )
            if result.returncode != 0 or not result.stdout.strip():
                self.console.print(
                    "[yellow]Warning: No models found. Starting Ollama...[/yellow]"
                )
                result = subprocess.run(
                    ["ollama", "serve"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
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

        # Check if this is a base model (no base_models) or derived model
        # base_models is None/empty for base models, populated for derived models
        is_base_model = getattr(model_info, "base_models", None) is None or (
            getattr(model_info, "base_models", None) == []
        )

        # Clean up the name for Ollama (replace special characters)
        # Drop the repo convention marker "-GGUF" (ollama accepts a plain GGUF
        # model id anyway).
        import re

        base_name = re.sub(r"-GGUF$", "", base_name)
        clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", base_name)[:35]

        # Add org prefix for base models, skip for derived models
        # Exception: if base name already starts with org, don't duplicate
        org_prefix = ""
        if is_base_model:
            org_name = parts[0]
            # Check if base_name already starts with org_name
            if not base_name.lower().startswith(org_name.lower()):
                org_prefix = f"{org_name}-"

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
        model_name = f"{org_prefix}{clean_name}{version_suffix}{context_suffix}"

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
            "num_thread": params.get("num_thread", 4),
        }

        # Inject num_thread / num_gpu from the active hardware profile
        params_dict = self._apply_hardware(params_dict)

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
                    "num_thread": int(input("Number of threads: ") or "4"),
                }

                # Inject num_thread / num_gpu from the active hardware profile
                params = self._apply_hardware(params)

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
  - jackrong/qwen3.5 -> qwen3.5-q3k_m
  - hf-coherence/command-r-plus -> command-r-plus
  - openbmb/minicpm5-2b -> minicpm5-2b-f16

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


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code.
    """
    cli = CLI()

    args = cli.parse_args()
    if args is None:
        return 0

    return cli.run(args)


if __name__ == "__main__":
    sys.exit(main())
