# Contribution Rules

This document defines how commits should be named and which ones appear in the changelog.

Consistent commit naming keeps release notes clear and meaningful for users, while avoiding noise from internal work.

## Commit Message Format

Each commit message starts with one of the keywords below (all lowercase), followed by a short description in plain English.

All commit type prefixes must be lowercase.

### Commit Types

| Type | Description | Changelog |
|------|-------------|-----------|
| `feat:` | New features or major improvements visible to users | ✓ |
| `fix:` | Bug fixes — things that were broken now behave correctly | ✓ |
| `change:` | Non-breaking behavior or UX changes users will notice | ✓ |
| `dataset:` | Changes to bundled datasets, prompts, or test seeds | ✓ |
| `dev:` | Internal maintenance, refactors, or tooling changes with no user-facing impact | ✗ |
| `docs:` | Documentation changes — READMEs, examples, or guides | ✓ |

### Examples

- `feat: add --auto-resume flag`
- `fix: avoid crash when results folder is missing`
- `change: enable language matching by default`
- `dataset: update base_user_inputs.jsonl`
- `dev: refactor runner for simplicity`
- `docs: clarify README usage examples`

---

## Pull Requests

- PR titles must follow the same `type: description` format.
- All PRs should be made to the `develop` branch.

## General Rules

- Keep commit messages **under 80 characters**.
- Use **imperative mood** (`add`, `fix`, `update`, not `added` or `fixed`).
- Write clear, self-contained commits — one logical change per commit.
- Avoid merging work with vague messages like "misc updates" or "wip".
