<!--
SPDX-FileCopyrightText: 2026 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Package manager**: `uv` (managed via `mise`)

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_lateimport.py

# Run a single test by name
uv run pytest tests/test_lateimport.py::TestLateImportResolution::test_resolves_module

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Build the package
uv run hatchling build

# REUSE license compliance check
reuse lint
```

## Architecture

This is a single-file stdlib-only package: **everything lives in `src/lateimport/__init__.py`**. There are no submodules.

### Two public tools

**`LateImport[T]` / `lateimport()`** — An explicit proxy for application code. Wraps a module import (and optional attribute chain) and defers resolution until the proxy is first *used* (called, subscripted, passed to `isinstance`, etc.). Uses `__slots__` and a `threading.Lock` for thread-safe double-checked resolution. Attribute access on an unresolved proxy returns a new child `LateImport` rather than resolving — only actual *use* triggers import.

**`create_late_getattr()`** — A factory for package `__init__.py` files using a `_dynamic_imports` dispatch table. Returns a `__getattr__` function that imports on demand and caches results directly into the module's `globals()`. The dispatch tuple `(package, target_module)` has a special case: `target_module == "__module__"` imports the submodule itself as the attribute, while any other value imports `package.target_module` and does `getattr(module, attr_name)`.

**`INTROSPECTION_ATTRIBUTES`** — A `frozenset` of dunder names (e.g. `__doc__`, `__name__`) that resolve the proxy immediately rather than returning a child proxy. This prevents introspection machinery (pydantic, `inspect`, dataclasses) from getting a proxy where a real value is expected.

### Key implementation detail

`LateImport` uses `object.__getattribute__` and `object.__setattr__` throughout to bypass its own `__getattr__`/`__setattr__` overrides when accessing internal state (`_module_name`, `_attrs`, `_resolved`, `_lock`, `_parent`). This is intentional — do not simplify these calls to `self._attr`.

## Tests

Tests are in `tests/`. `test_lateimport.py` is the primary test suite; `test_late_import_more.py` contains additional scenarios and edge cases.

pytest marks used: `unit`, `benchmark`, `performance` (no custom plugins required; these are informational).

## License

Dual-licensed `MIT OR Apache-2.0`. All files must carry SPDX headers. Use `reuse lint` to verify compliance before commits.
