"""
Modelfile generation and validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich.console import Console


class ModelfileGenerator:
    """Generates and validates Modelfile content."""
    
    def __init__(self):
        """Initialize the Modelfile generator."""
        self.console = Console()
    
    def generate(self, model_id: str, context_length: int, params: Dict[str, Any]) -> str:
        """Generate Modelfile content from a model ID and parameters.
        
        Args:
            model_id: The Hugging Face model identifier.
            context_length: The context window size in tokens.
            params: Dictionary of parameters (temperature, top_p, num_threads, etc.).
            
        Returns:
            Generated Modelfile content.
        """
        from .templates import TemplateManager
        
        template_manager = TemplateManager()
        
        # Try to load a template first
        template = template_manager.load_template("default")
        if template is None:
            self.console.print("[yellow]Warning: No default template found. Using basic format.[/yellow]")
            template = self._basic_template(model_id, context_length, params)
        
        # Render the template
        context = {
            "model_id": model_id,
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.9),
            "context_length": context_length,
            "num_threads": params.get("num_threads", 4),
        }
        
        content = template_manager.render(template, context)
        return content
    
    def _basic_template(self, model_id: str, context_length: int, params: Dict[str, Any]) -> str:
        """Generate a basic Modelfile without template.
        
        Args:
            model_id: The model identifier.
            context_length: Context window size.
            params: Additional parameters.
            
        Returns:
            Basic Modelfile content.
        """
        lines = [
            f"FROM {model_id}",
            "",
            "# Parameters",
            f"PARAMETER temperature {params.get('temperature', 0.7)}",
            f"PARAMETER top_p {params.get('top_p', 0.9)}",
            f"PARAMETER num_predict {context_length}",
            f"PARAMETER num_threads {params.get('num_threads', 4)}",
        ]
        return "\n".join(lines)
    
    def validate_modelfile(self, content: str) -> bool:
        """Validate Modelfile syntax.
        
        Args:
            content: The Modelfile content to validate.
            
        Returns:
            True if valid, False otherwise.
        """
        lines = content.split("\n")
        
        errors = []
        param_errors = []
        
        # Check for FROM statement
        has_from = False
        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith("FROM "):
                has_from = True
                break
        
        if not has_from:
            errors.append("Modelfile must contain a FROM statement")
        
        # Check for valid PARAMETER lines
        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith("PARAMETER "):
                parts = stripped.split(None, 1)
                if len(parts) < 2:
                    param_errors.append(f"Invalid PARAMETER line: {stripped}")
                else:
                    param_name, param_value = parts
                    # Validate parameter values
                    valid_values = ["true", "false", "0", "1"]
                    if param_value not in valid_values:
                        try:
                            # Try to parse as number (int or float)
                            float(param_value)
                        except ValueError:
                            # String value - check for common patterns
                            if not any(param_value.startswith(prefix) for prefix in [
                                "<s>", "</s>", "<|eot_id|>", "<|eom_id|>", "<|eop_id|>",
                                "<|start_header_id|>", "<|end_header_id|>", "<|user|>",
                                "<|assistant|>", "<|system|>", "<|observation|>",
                                "<|function|>", "<|python_call|>", "<|python_end|>",
                                "</summary>", "</thinking>", "</analysis>", "<|end_of_text|>",
                                "<|eom_id|>", "<|eop_id|>", "<|start_header_id|>",
                                "<|end_header_id|>", "<|user|>", "<|assistant|>",
                                "<|system|>", "<|observation|>", "<|function|>",
                                "<|python_call|>", "<|python_end|>",
                                "1024x1024", "1792x1024", "1024x1792",
                            ]):
                                param_errors.append(f"Unknown parameter value: {param_name}={param_value}")
        
        if param_errors:
            errors.extend(param_errors)
        
        if errors:
            self.console.print("[red]Validation errors:[/red]")
            for error in errors:
                self.console.print(f"  - {error}")
            return False
        
        self.console.print("[green]Modelfile is valid.[/green]")
        return True
    
    def save(self, content: str, filepath: str) -> bool:
        """Save Modelfile content to a file.
        
        Args:
            content: The Modelfile content to save.
            filepath: The path to save the file.
            
        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            self.console.print(f"[green]Modelfile saved to: {filepath}[/green]")
            return True
        except Exception as e:
            self.console.print(f"[red]Error saving file: {e}[/red]")
            return False
