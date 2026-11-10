# AILuminate English Demo

Imports the public [MLCommons AILuminate v1.0 Demo](https://github.com/mlcommons/ailuminate):
1,200 English prompts across twelve hazard categories. No account or token is required.
The source CSV contains both direct and adversarial requests.

From your Spikee workspace:

```bash
python datasets/seeds-ailuminate-demo-en-us/fetch_and_convert_dataset.py
spikee generate --seed-folder datasets/seeds-ailuminate-demo-en-us \
  --include-standalone-inputs
```

The fetcher checks a pinned upstream revision and SHA-256 before writing any files.
It retains all 1,200 original records in `source.csv` and records the source, license,
checksum and selected counts in `source_manifest.json`. The default `standalone_user_inputs.jsonl`
contains **1,100 prompts**, excluding Specialized Advice. Re-running with
`--include-specialized-advice` includes all 1,200 prompts and replaces the converted file.
Use `--source-csv path/to/source.csv` to convert a verified local copy without downloading.

Each source prompt is stored unchanged in the standalone record's `text` field.
The base-input, jailbreak and instruction files remain empty, following the same
pattern as the WildGuard and Simsonsun imports. Apply plugins with `--plugins`
when generating, and choose attacks normally when testing.

## Assessment

Entries select `llm_judge_ailuminate`, an independent LLM judge using the published
[AILuminate Assessment Specification v1.4](https://drive.google.com/file/d/1Kh3G39PXhai_Lk6NLWL5OjWbcgDD-KFW/view).
Select its model with the normal `--judge-options provider/model` option.
The original seed, source ID, source hash, persona, locale and revision survive
generation in `judge_args`, including when a plugin changes the input.

The judge scores legitimization and enablement separately, then applies the category's
violation rule. Privacy and Sexual Content use enablement alone. Specialized Advice
allows disclaimed advice, unlike Spikee's investment-advice guardrail dataset.
Other hazards in the same response are still assessed. Normal results save a boolean;
the component scores are not retained. Invalid assessments receive one format-repair
retry; provider errors and exhausted repairs remain errors.

This is Demo dataset support with an independent evaluator, not the private MLCommons
jailbreak corpus, official evaluator or official benchmark grading. The judge has
offline policy/parser tests; its semantic accuracy has not been measured against
human-labelled AILuminate responses.

## Attribution and license

Prompt data: **MLCommons AI Risk & Reliability working group**, AILuminate v1.0 Demo,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated in the
[upstream dataset README](https://github.com/mlcommons/ailuminate#license).
Conversion changes the file format and optionally excludes Specialized Advice;
prompt text is preserved. Retain this attribution and the license link when sharing
the converted data. Source prompts and converted prompts are downloaded locally,
not bundled with Spikee.
