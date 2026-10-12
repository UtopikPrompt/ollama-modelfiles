# Virtual Environment Management

## Creating Virtual Environment

```bash
# Create venv using python
python -m venv venv

# Create venv using virtualenv (if installed)
virtualenv venv
```

## Activating Virtual Environment

```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

## Managing Dependencies

```bash
# Install from pyproject.toml
pip install -e .

# Install only production dependencies
pip install -e ".[prod]"

# Install dev dependencies
pip install -e ".[dev]"

# Upgrade all packages
pip install --upgrade -r requirements.txt
```

## Deactivating Virtual Environment

```bash
deactivate
```

## Useful Commands

```bash
# Check Python version
python --version

# Check pip version
pip --version

# List installed packages
pip list

# Show package metadata
pip show requests

# Upgrade a specific package
pip install --upgrade requests

# Show what's in requirements.txt
pip freeze
```

## Common Issues

### Multiple Python Versions
```bash
# Check available Python versions
py -l

# Use specific Python version
python3.11 -m venv venv
```

### venv Not Found
```bash
# Make sure virtualenv is installed
pip install virtualenv

# Create venv again
python -m venv venv
```

### Activation Not Working
```bash
# Check path
where venv\Scripts\activate

# Add to PATH in PowerShell
$env:Path + ";C:\path\to\venv\Scripts"
```
