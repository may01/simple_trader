# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ompytr** is a Python-based algorithmic trading system. This is an early-stage project focused on building trading strategies, market data handling, and execution infrastructure.

## Development Setup

### Python Environment
- **Python version**: 3.9+ (recommended 3.11+)
- **Package management**: Use `pip` with `requirements.txt`
- **Virtual environment**: Create with `python -m venv venv` and activate before development
- **Dependencies**: Core dependencies should be minimal; separate `requirements-dev.txt` for development/testing tools

### Common Commands

Once the project is set up with package structure:

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests (pytest assumed)
pytest

# Run specific test
pytest tests/test_specific.py::test_function

# Type checking (mypy recommended)
mypy src/

# Linting
flake8 src/ tests/
black src/ tests/

# Format code
black --check src/ tests/
```


## Code Guidelines

- Use type hints for all function signatures (enables static analysis and better IDE support)
- Strategies should be stateless or have clear state management
- Market data handlers should normalize different sources to a common interface
- Position/portfolio state should be immutable when possible (use dataclasses)
- Order execution should be atomic and logged

## Development Workflow

- Create feature branches for new strategies or components
- Each component should have corresponding tests before or alongside implementation
- Run type checking and linting before committing
