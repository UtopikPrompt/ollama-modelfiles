"""
Template management for Modelfile generation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from jinja2.sandbox import SandboxedEnvironment


class TemplateManager:
    """Manages template loading and rendering."""
    
    def __init__(self):
        """Initialize the template manager."""
        # Use SandboxedEnvironment for security
        self._env = SandboxedEnvironment(
            loader=FileSystemLoader("templates/jinja2"),
            undefined=StrictUndefined,
        )
        self._templates: Dict[str, str] = {}
    
    def load_template(self, name: str) -> Optional[str]:
        """Load a template by name.
        
        Args:
            name: The template name (e.g., 'chat', 'completion', 'image', 'default').
            
        Returns:
            Template source code or None if not found.
        """
        # First try jinja2 templates
        if name in self._templates:
            return self._templates[name]
        
        # Try to load from jinja2 directory
        try:
            template = self._env.get_template(f"{name}.j2")
            return template.source
        except jinja2.TemplateNotFound:
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
            template: The template source code.
            context: Dictionary of template variables.
            
        Returns:
            Rendered template string.
        """
        try:
            # Use the same SandboxedEnvironment for consistency
            template_obj = self._env.from_string(template)
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
                        templates[filename[:-4]] = open(os.path.join(template_dir, filename), "r", encoding="utf-8").read()
        except Exception:
            pass
        
        return templates
