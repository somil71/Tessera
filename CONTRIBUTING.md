# Contributing to Tessera

Thank you for your interest in contributing to Tessera!

## Development Setup

1. **Rust Toolchain:** Ensure you have a recent stable Rust toolchain (2024 edition support).
2. **Python Environment:** Create a virtual environment with Python 3.11 and install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
   ```
3. **Build:**
   ```powershell
   cargo build
   ```

## Testing Guidelines

Before opening a pull request, ensure all test suites pass:

1. **Rust Unit Tests & Linter:**
   ```powershell
   cargo test
   cargo clippy --all-targets -- -D warnings
   ```
2. **Python Smoke Tests:**
   ```powershell
   .\.venv\Scripts\python.exe tests/fixtures/smoke_job_memory.py
   .\.venv\Scripts\python.exe tests/fixtures/smoke_auto_raster.py
   ```

## Code Review & Memory Constraints

- Tessera relies on strict committed-memory caps enforced by Windows Job Objects.
- Any new execution path must handle memory budget enforcement and preserve georeferencing tags and CRS.
