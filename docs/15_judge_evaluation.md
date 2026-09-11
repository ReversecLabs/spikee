# Test judge correctness

Run from the repository root with Spikee and pytest installed. The suite uses fixed labelled responses; it never calls a target or generates attacks.

## Offline: no API calls

```bash
SPIKEE_TESTS_USE_ISOLATED_VENV=false \
  python -m pytest tests/functional/test_judge_evaluation.py -q
```

Covers all five judges: verdict parsing, invalid scores, retries, literal matches, dataset generation and standard/LLMJailbreaker/Crescendo judge calls.

## Live: start with one case

Put `OPENROUTER_API_KEY` in `workspace/.env`, or set `SPIKEE_TEST_ENV_FILE` to another file. Variables load into memory; existing environment values win.

```bash
RUN_JUDGE_EVALS=1 SPIKEE_TESTS_USE_ISOLATED_VENV=false \
  python -m pytest tests/inference/judges -q -s \
  --judge-model=openrouter/qwen/qwen3.8-flash \
  --judge-case=incident-language-refusal --judge-repeats=3 \
  --judge-report=/tmp/judges-refusal.jsonl
```

Only `RUN_JUDGE_EVALS=1` enables this suite. A key alone or a flag set to `0` does not. The general inference suite uses the separate `RUN_INFERENCE_TESTS=1` gate. An enabled run fails if credentials are missing.

## Run all models

```bash
RUN_JUDGE_EVALS=1 SPIKEE_TESTS_USE_ISOLATED_VENV=false \
  python -m pytest tests/inference/judges -q -s \
  --judge-repeats=3 --judge-workers=5 \
  --judge-phase=baseline --judge-report=/tmp/judges-baseline.jsonl
```

Default OpenRouter models (each prefixed with `openrouter/`):

```text
deepseek/deepseek-v4-flash-0731
qwen/qwen3.8-flash
qwen/qwen3.8-27b
qwen/qwen3.6-35b-a3b
z-ai/glm-5.3-flash
```

This schedules **735 evaluations**: 49 cases × 5 models × 3 repetitions. Empty target responses skip inference; format repair can add a call. `SPIKEE_API_TIMEOUT` defaults to 90 seconds; provider retries can extend total runtime.

| Option | Use |
|---|---|
| `--judge-model=provider/model` | Repeat to select several models. |
| `--judge-case=criteria-refusal` | Filter case IDs by substring. |
| `--judge-split=development` | Tune prompts on development cases. Use `holdout` for validation. |
| `--judge-report=/tmp/judges-candidate.jsonl` | Use a new path for each run; existing files are rejected. |

## Compare

Run the same cases/models/repetitions before and after edits, changing the report path and phase. Then:

```bash
python -m tests.judge_report /tmp/judges-baseline.jsonl /tmp/judges-candidate.jsonl
```

Reports retain expected/actual verdicts, prompts, raw replies, source hashes, latency and available usage/cost metadata. Rows flush as they finish. False positives, false negatives, invalid verdicts and provider errors stay separate; any failure makes pytest fail **after all cases run**. Keep unsuccessful reports and check positives as well as refusals.

## Cases

[`tests/fixtures/judge_cases.jsonl`](../tests/fixtures/judge_cases.jsonl) holds 60 labelled cases: 49 LLM and 11 literal. They cover harmful, investment, disclosure and objective tasks, including refusals, compliance, misleading disclaimers, multilingual answers and evaluator injection.

Entries use the production generator and `call_judge(entry, response)`. Instruction seeds use the direct/no-jailbreak wrapper; standalone seeds use their normal generator. Source judge/arguments are preserved, except explicit objective-judge overrides because no bundled seed selects that judge. Development and holdout source objectives are separate within each judge.

See [recorded results](./16_judge_evaluation_results.md). This is a regression corpus, not a broad accuracy benchmark. Rejudging and attack-specific refusal/backtracking helpers are outside its scope.
