# Functional Testing Guide

Spikee ships with an end-to-end functional suite that exercises the CLI exactly the way a user would. The tests live under `tests/functional` and currently focus on verifying `spikee generate` across the key combinations of flags, plugins, and seed data. This document explains how to run them locally while you are iterating on the codebase.

## 1. Prerequisites

- Python 3.11 or later (check `python --version`).
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
   - Bootstrap a scratch workspace (`spikee init`) and overlay the fixtures under `tests/functional/workspace`.
   - Execute the relevant `spikee` CLI commands (currently `spikee generate` and `spikee test`) and assert the outputs.

   Optionally, run the functional suite from your workspace `pytest ../tests/functional` to use enviromental variables from your workspace .env file:
   - `SPIKEE_TESTS_USE_ISOLATED_VENV=false` - Uses the current environment instead of creating a new one for each test session. Use `true` (the default) for an isolated environment.
   - Uses LLM provider inference keys.

4. **Run a single test** (useful while debugging):

   ```bash
   pytest tests/functional/test_judge_evaluation.py
   ```

   Add `-k <substring>` or `-vv` for tighter filtering or more verbose logs.

## 3. Notes & Troubleshooting

- **Judge evaluations**: See [Evaluating judge correctness](./15_judge_evaluation.md) for dataset-derived offline regressions and the separately gated OpenRouter evaluation matrix.

- **Temporary workspaces**: Every test uses Pytest’s `tmp_path` fixture. Nothing under your repo or existing workspaces is modified.
- **Dependencies**: The fixture executes `pip install .` inside its temporary venv, so the full dependency tree is downloaded automatically. Make sure you have network access (or a package mirror) the first time you run the suite.
- **Matrix coverage**: Many tests are parameterised to run with `--match-languages` enabled and disabled. Expect the runtime to roughly double because both code paths are executed.
- **Debugging outputs**: Use `pytest -s` to keep stdout/stderr from the CLI if you are investigating a failure.

## 4. Next Steps

- Mirror this pattern for `spikee test` workflows to cover targets, judges, and attacks.
- Wire the functional suite into CI so releases fail fast if a regression slips in.

That’s it—run `pytest tests/functional` anytime you need confidence that the end-to-end CLI behaviour remains intact. Happy hacking!

## Live provider tests

Live tests incur provider API charges and require explicit opt-in:

```bash
RUN_INFERENCE_TESTS=1 SPIKEE_TESTS_USE_ISOLATED_VENV=false \
  python -m pytest tests/inference/test_inference.py tests/inference/test_workspace_inference.py -v
```

Credentials load from `workspace/.env`; use `SPIKEE_TEST_ENV_FILE` for another
path. Existing environment variables take precedence. Provider errors, billing
failures and malformed replies are failures, not successful connectivity checks.
An attack model's explicit refusal is accepted only when its reply provides
that evidence; a JSON parsing error alone is insufficient.

The smoke-test model IDs are centralized in `tests/inference/models.py`.
Override an ID with `SPIKEE_TEST_<PROVIDER>_MODEL`, for example
`SPIKEE_TEST_OPENROUTER_MODEL=qwen/qwen3.8-flash`. Use the model ID without the
Spikee provider prefix. Azure live cases are enabled only when
`SPIKEE_TEST_AZURE_OPENAI_MODEL` names a deployment on the configured resource;
Azure's public model catalog does not establish deployment availability.
The Azure adapter uses the v1 API; remove dated `AZURE_OPENAI_API_VERSION` /
`OPENAI_API_VERSION` settings or set them to `v1`.

Groq cases are spaced 65 seconds apart to accommodate small token-per-minute
quotas. Set `SPIKEE_TEST_GROQ_INTERVAL` to match the account's quota. This does not
reduce the number of attack iterations or turn rate-limit failures into passes.
The multilingual Groq case also sets `max-tokens=480` per reply to stay within
the small output-token quota; the attack leaves this limit unset by default.
The separately gated judge-quality evaluation matrix is documented in
[Evaluating judge correctness](15_judge_evaluation.md).
