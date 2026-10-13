"""
Template management for Modelfile generation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import jinja2
from jinja2 import Environment, FileSystemLoader, StrictUndefined, Undefined
from jinja2.sandbox import SandboxedEnvironment


class TemplateManager:
    """Manages template loading and rendering."""

    def __init__(self):
        """Initialize the template manager."""
        # Use SandboxedEnvironment for security
        # The loader will look relative to the package location
        import os

        script_dir = os.path.dirname(os.path.abspath(__file__))
        jinja2_dir = os.path.join(script_dir, "templates", "jinja2")
        self._env = SandboxedEnvironment(
            loader=FileSystemLoader(jinja2_dir),
            undefined=Undefined,
        )
        self._templates: Dict[str, str] = {}

    def load_template(self, name: str) -> Optional[str]:
        """Load a template by name.

        Args:
            name: The template name (e.g., 'chat', 'completion', 'image', 'default').

        Returns:
            Rendered template string or None if not found.
        """
        # First try jinja2 templates - render them immediately with proper macro support
        if name in self._templates:
            return self._templates[name]

        # Try to load and render from jinja2 directory
        try:
            template = self._env.get_template(f"{name}.j2")
            # Handle both old and new Jinja2 API
            try:
                # Newer Jinja2 (3.1+) uses render() method
                return template.render()
            except AttributeError:
                # Older Jinja2 uses .source attribute
                template_obj = self._env.from_string(template.source)
                return template_obj.render()
        except jinja2.TemplateNotFound:
            pass
        except jinja2.TemplateSyntaxError:
            # Template has syntax errors (e.g., undefined macros)
            pass
        except Exception:
            pass

        # Try to load from text templates
        try:
            # Get the directory where this script is located
            import os

            script_dir = os.path.dirname(os.path.abspath(__file__))
            template_path = os.path.join(script_dir, "templates", f"{name}.txt")
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def render(self, template: str, context: Dict[str, Any]) -> str:
        """Render a template with the given context.

        Args:
            template: The template source code (already rendered templates passed through as-is).
            context: Dictionary of template variables.

        Returns:
            Rendered template string.
        """
        try:
            # If template is already rendered, return it as-is
            if "\n" not in template or "FROM " in template and "PARAMETER" in template:
                return template

            # Create a new environment with the same configuration for rendering
            # This allows macros to be resolved properly
            render_env = SandboxedEnvironment(
                loader=self._env.loader,
                undefined=self._env.undefined,
            )
            # from_string() uses the environment's loader, not a passed one
            template_obj = render_env.from_string(template)
            return template_obj.render(**context)
        except Exception as e:
            print(f"Error rendering template: {e}")
            return str(e)

    def get_templates(self) -> List[str]:
        """Get list of available templates.

        Returns:
            List of template names.
        """
        templates = []

        # Try jinja2 templates
        try:
            for filename in self._env.list_templates():
                name = filename.replace(".j2", "")
                templates.append(name)
        except Exception:
            pass

        # Try text templates
        try:
            import os

            script_dir = os.path.dirname(os.path.abspath(__file__))
            template_dir = os.path.join(script_dir, "templates")
            if os.path.exists(template_dir):
                for filename in os.listdir(template_dir):
                    if filename.endswith(".txt"):
                        templates.append(filename[:-4])
        except Exception:
            pass

        return sorted(set(templates))

    def load_all_templates(self) -> Dict[str, str]:
        """Load all available templates.

        Returns:
            Dictionary mapping template names to their source code.
        """
        templates = {}

        # Load jinja2 templates
        try:
            for template in self._env.list_templates():
                name = template.replace(".j2", "")
                templates[name] = self._env.get_template(template).source
        except Exception:
            pass

        # Load text templates
        try:
            import os

            script_dir = os.path.dirname(os.path.abspath(__file__))
            template_dir = os.path.join(script_dir, "templates")
            if os.path.exists(template_dir):
                for filename in os.listdir(template_dir):
                    if filename.endswith(".txt"):
                        templates[filename[:-4]] = open(
                            os.path.join(template_dir, filename), "r", encoding="utf-8"
                        ).read()
        except Exception:
            pass

        return templates
