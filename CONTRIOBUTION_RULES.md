# Contribution Rules

This document defines how commits should be named and which ones appear in the changelog.

Consistent commit naming keeps release notes clear and meaningful for users, while avoiding noise from internal work.

---

## Commit Message Format

Each commit message starts with one of the keywords below, followed by a short description in plain English.

Example:
```

feat: add --auto-resume flag for interrupted tests
fix: avoid crash when results folder is missing
change: rename --standalone-attacks to --include-standalone-inputs
dataset: refresh base_user_inputs.jsonl with updated prompts
dev: refactor result parser for readability
docs: clarify README usage examples

```

---

## Commit Types

### `feat:`  
New features or major improvements visible to users.  
Examples:
- `feat: add --auto-resume flag`
- `feat: improve CLI progress display`

### `fix:`  
Bug fixes — things that were broken now behave correctly.  
Examples:
- `fix: avoid crash when results folder is missing`
- `fix: correct default output path`

### `change:`  
Non-breaking behavior or UX changes that users will notice.  
Examples:
- `change: enable language matching by default`
- `change: rename CLI flag for clarity`

### `dataset:`  
Changes to bundled datasets, prompts, or test seeds that affect tool behavior or evaluation results.  
Examples:
- `dataset: update base_user_inputs.jsonl`
- `dataset: correct mislabeled input samples`

### `dev:`  
Internal maintenance, refactors, or cosmetic changes with no user-facing impact.  
Examples:
- `dev: refactor runner for simplicity`
- `dev: rename variables for clarity`

### `docs:`  
Documentation changes — READMEs, examples, or guides.  
Examples:
- `docs: update usage examples`
- `docs: fix changelog typo`

---

## Changelog Inclusion

Commits included in **CHANGELOG.md**:
```

feat, fix, change, dataset

```

Commits excluded from the changelog:
```

dev, docs

```

---

## General Rules

- Keep commit messages **under 80 characters**.
- Use **imperative mood** (`add`, `fix`, `update`, not `added` or `fixed`).
- Write clear, self-contained commits — one logical change per commit.
- Avoid merging work with vague messages like “misc updates” or “wip”.


