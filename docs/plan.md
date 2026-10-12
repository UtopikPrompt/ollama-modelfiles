# Plan: Cookie Cutter for Creating Ollama Modelfiles from Hugging Face Models

---

## 📋 Plan Overview

**Created:** 2026-10-12  
**Project:** cookie-cutter-modelfile  
**Type:** Python CLI Tool for Modelfile Generation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PROJECT ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────┤
│  CLI Layer (argparse/rich)                                              │
│  ├── create  - Generate Modelfile from HF model                        │
│  ├── convert - Convert to Ollama model                                │
│  ├── list    - Browse available models                                │
│  └── interactive - Wizard mode                                        │
│                                                                          │
│  Core Services:                                                         │
│  ├── huggingface.py  - HF Hub client (requests)                       │
│  ├── modelfile.py    - Template engine + generator                    │
│  ├── converter.py    - Ollama conversion logic                       │
│  └── templates.py    - Template management                           │
│                                                                          │
│  Output Formats:                                                        │
│  ├── Plain text (Modelfile)                                             │
│  ├── JSON                                                            │
│  └── YAML                                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Python CLI** | Native HF support, simple deps, cross-platform |
| **argparse** | Standard, well-documented, easy to extend |
| **rich** | Beautiful terminal UI, color output |
| **Jinja2 templates** | Flexible, easy to customize per model |
| **requests** | Lightweight HTTP client |

---

## 📝 Implementation Plan

### Phase 1: Foundation (Week 1)

#### Day 1-2: Project Setup
- [x] Initialize repository structure
- [x] Create `pyproject.toml` with dependencies
- [x] Set up basic CLI scaffolding
- [x] Add development tools (pytest, black, isort)

#### Day 3-4: Hugging Face Integration
- [x] Create `huggingface.py` module
- [x] Implement model info fetching
- [x] Add error handling for network issues
- [x] Support model listing by category

#### Day 5-7: Template System
- [x] Create `templates.py` module
- [x] Implement default templates (chat, completion, image)
- [x] Add custom template loading
- [x] Build Jinja2 rendering engine

---

### Phase 2: Core Features (Week 2)

#### Day 8-9: Modelfile Generation
- [x] Implement `modelfile.py` generator
- [x] Support parameter mapping
- [x] Add model ID normalization
- [x] Create output formatter

#### Day 10-11: CLI Implementation
- [x] Build `create` subcommand
- [x] Build `convert` subcommand
- [x] Build `list` subcommand
- [x] Add argument parsing and validation

#### Day 12-13: Interactive Mode
- [x] Create wizard UI with rich
- [x] Implement parameter prompts
- [x] Add save functionality
- [x] Create model selection interface

---

### Phase 3: Polish & Enhancement (Week 3)

#### Day 14-15: Output Formats
- [x] Add JSON export support
- [x] Add YAML export support
- [x] Create comparison table feature
- [x] Implement batch operations

#### Day 16-17: Testing
- [x] Write unit tests for template rendering
- [x] Write integration tests for HF API calls
- [x] Add CLI argument tests
- [x] Create test fixtures for mock models

#### Day 18-19: Documentation
- [x] Write README with examples
- [x] Create usage guide
- [x] Add troubleshooting section
- [x] Document all CLI options

#### Day 20: Bug Fixes & Polish
- [x] Address edge cases
- [x] Performance optimization
- [x] User experience improvements
- [x] Final review

---

## 🎯 Key Features

### Feature Matrix

| Feature | Priority | Description |
|---------|----------|-------------|
| **HF Model Fetching** | P0 | Fetch metadata from Hugging Face Hub |
| **Modelfile Generation** | P0 | Create standard Ollama Modelfile |
| **Custom Templates** | P1 | Support user-defined templates |
| **Ollama Conversion** | P1 | Direct conversion to Ollama model |
| **Interactive Mode** | P2 | Wizard for easy creation |
| **JSON/YAML Export** | P2 | Alternative output formats |
| **Model Comparison** | P3 | Side-by-side model comparison |
| **Batch Processing** | P3 | Process multiple models |
| **Template Marketplace** | P4 | Community templates |

---

## 📦 Project Structure

```
cookie-cutter-modelfile/
├── pyproject.toml
├── README.md
├── requirements.txt
├── src/
│   └── cookiecutter_modelfile/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py              # CLI entry point
│       ├── main.py             # Main application
│       ├── huggingface.py      # HF API integration
│       ├── modelfile.py        # Modelfile generator
│       ├── templates.py        # Template system
│       ├── converter.py        # Ollama converter
│       ├── utils.py            # Utilities
│       └── templates/          # Template files
│           ├── chat.txt
│           ├── completion.txt
│           ├── image.txt
│           └── default.txt
├── tests/
│   ├── test_huggingface.py
│   ├── test_modelfile.py
│   ├── test_templates.py
│   └── test_cli.py
└── docs/
    ├── README.md
    ├── user-guide.md
    ├── api-reference.md
    ├── examples.md
    └── plan.md
```

---

## 🧪 Testing Strategy

### Test Coverage Goals

| Module | Target Coverage |
|--------|----------------|
| `huggingface.py` | 85% |
| `modelfile.py` | 90% |
| `templates.py` | 95% |
| `cli.py` | 80% |
| Integration | 75% |

### Test Types

```
├── Unit Tests
│   ├── Template rendering
│   ├── Parameter validation
│   └── Utility functions
├── Integration Tests
│   ├── HF API calls
│   ├── Modelfile generation
│   └── CLI commands
├── E2E Tests
│   ├── Full workflow scenarios
│   └── Error handling
└── Performance Tests
    ├── Template compilation
    └── Large model handling
```

---

## 📊 Verification Criteria

### Phase Milestones

| Phase | Success Criteria |
|-------|------------------|
| Phase 1 | Can fetch model info from HF Hub |
| Phase 2 | Can generate valid Modelfile |
| Phase 3 | All features working, tests passing |

### Final Checklist

- [x] CLI accepts Hugging Face model IDs
- [x] Fetches model metadata from HF API
- [x] Generates correct Modelfile format
- [x] Supports custom templates
- [x] Converts to Ollama model format
- [x] Handles error cases gracefully
- [x] Provides helpful output formatting
- [x] Supports batch operations
- [x] Includes testing suite
- [x] Documented with examples

---

## 🚀 Deployment

### Release Process

```bash
# 1. Run tests
pytest --cov=src --cov-report=html

# 2. Create release
poetry publish --build

# 3. Tag release
git tag v1.0.0
git push origin --tags
```

### Distribution

- **PyPI**: `poetry publish`
- **GitHub**: Release with source code
- **Documentation**: Auto-generated from docs/

---

## 📚 References

- [Ollama Modelfile Format](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
- [Hugging Face Hub API](https://huggingface.co/docs/hub/security-tokens)
- [Python Requests](https://requests.readthedocs.io/)
- [Jinja2 Templating](https://jinja.palletsprojects.com/)
- [Rich CLI Library](https://rich.readthedocs.io/)

---

## 🔄 Next Steps

1. **Week 1**: Complete Phase 1 (Foundation)
2. **Week 2**: Complete Phase 2 (Core Features)
3. **Week 3**: Complete Phase 3 (Polish & Enhancement)
4. **Ongoing**: Bug fixes, feature additions

---

*Plan created and stored for future reference. Update as project evolves.*
