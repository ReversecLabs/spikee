# Functional Testing Guide

Spikee ships with an end-to-end functional suite that exercises the CLI exactly the way a user would. The tests live under `tests/functional` and currently focus on verifying `spikee generate` across the key combinations of flags, plugins, and seed data. This document explains how to run them locally while you are iterating on the codebase.

## 1. Prerequisites

- Python 3.9 or later (check `python --version`).
- A local checkout of this repository.
- `pytest` available in your environment (install it once with `pip install pytest` if you do not already have it).

## 2. Recommended workflow

1. **Create and activate a virtual environment** (only needed once per clone):

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```

2. *(Optional)* **Install Spikee locally.** The functional harness performs a full `pip install .` inside its own temporary venv, so pre-installing is not required. If you want an editable install for day-to-day hacking, run:

   ```bash
   pip install -e .
   ```

3. **Run the functional suite** from the repository root:

   ```bash
   pytest tests/functional
   ```

   The harness will:

   - Spawn a temporary virtual environment for each session.
   - Install the current `spikee` wheel into that venv.
   - Bootstrap a scratch workspace (`spikee init`) and overlay the fixtures under `tests/functional/fixtures`.
   - Execute the relevant `spikee` CLI commands (currently `spikee generate` and `spikee test`) and assert the outputs.

4. **Run a single test** (useful while debugging):

   ```bash
   pytest tests/functional/test_generate_cli.py::test_generate_with_multiple_plugins_combines_results
   ```

   Add `-k <substring>` or `-vv` for tighter filtering or more verbose logs.

## 3. Notes & Troubleshooting

- **Temporary workspaces**: Every test uses Pytest’s `tmp_path` fixture. Nothing under your repo or existing workspaces is modified.
- **Dependencies**: The fixture executes `pip install .` inside its temporary venv, so the full dependency tree is downloaded automatically. Make sure you have network access (or a package mirror) the first time you run the suite.
- **Matrix coverage**: Many tests are parameterised to run with `--match-languages` enabled and disabled. Expect the runtime to roughly double because both code paths are executed.
- **Debugging outputs**: Use `pytest -s` to keep stdout/stderr from the CLI if you are investigating a failure.

## 4. Next Steps

- Mirror this pattern for `spikee test` workflows to cover targets, judges, and attacks.
- Wire the functional suite into CI so releases fail fast if a regression slips in.

That’s it—run `pytest tests/functional` anytime you need confidence that the end-to-end CLI behaviour remains intact. Happy hacking!
