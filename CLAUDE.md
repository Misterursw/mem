# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
请尽量保持中文回答与中文对话，注释也用中文。
### Development
- `uv run ruff format ./src` - Format Python code
- `uv run ruff check --fix ./src` - Auto-fix linting issues
- `uv run ruff format --check ./src` - Check formatting (CI)
- `uv run ruff check ./src` - Lint code (CI)

### Testing
- `uv run pytest` - Run all tests
- `uv run pytest -k "test_name"` - Run specific test
- `uv run pytest -vvv` - Run tests with verbose output
- `uv run pytest -n auto` - Run tests in parallel
- `make doctest` - Run docstring examples (starts LangGraph server automatically)
- `make doctest-watch` - Watch mode for docstring tests

### Documentation
- `make serve-docs` - Serve documentation locally
- `make build-docs` - Build documentation
- `make format-docs` - Format documentation Python code

## Architecture

LangMem is a Python library for AI agent memory management with these core modules:

### Core Components

1. **knowledge/** - Memory extraction and management
   - `create_memory_manager()` - Extract structured information from conversations
   - `create_memory_store_manager()` - Persist memories to LangGraph BaseStore
   - `create_manage_memory_tool()` / `create_search_memory_tool()` - Tools for agent memory operations
   - `create_thread_extractor()` - Generate conversation summaries

2. **prompts/** - Prompt optimization utilities
   - `create_prompt_optimizer()` - Single prompt improvement using metaprompt, gradient, or prompt memory strategies
   - `create_multi_prompt_optimizer()` - Multi-prompt optimization for consistency across prompt chains

3. **graphs/** - LangGraph integration utilities
   - Semantic graph operations
   - Authentication helpers

4. **short_term/** - Short-term memory and summarization

### Key Dependencies
- LangGraph (>=0.6.0) for graph-based workflows and storage
- LangChain for LLM abstractions
- Uses uv for dependency management and task running

### Memory Architecture
The library provides two approaches:
1. **Functional core** - Pure functions that work with any storage system
2. **LangGraph integration** - Native integration with LangGraph's Long-term Memory Store

Memory tools can be used "in the hot path" (agents actively manage memory during conversations) or "in the background" (automated memory management).