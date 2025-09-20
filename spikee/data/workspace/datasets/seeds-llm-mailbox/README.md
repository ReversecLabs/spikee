# Dataset

This is an example seeds folder to generate custom datasets for a specific use case (LLM Mailbox), as described in this article: https://labs.withsecure.com/tools/spikee. It's adapted from seeds-cybersec-2025-04 with instructions to target the specific user-case and intent, that isto steal a password reset token in the user's mailbox.

## English dataset
The English dataset can be generated using this command:

```bash
spikee generate --seed-folder datasets/seeds-llm-mailbox --languages en --format user-input
```

As we are not targeting an LLM but a GenAi feature, we need to use `--format user-input` (this is default from spikee 0.4.1) to ensure the generated payload does NOT include instructions but just the bare documents with the injected payloads.

## English dataset with delimiter bypass
If you have enabled prompt engineering defences in LLM Mailbox, you can create a dataset with custom delimiters to try and bypass the delimiters used in the prompt:

```bash
spikee generate --seed-folder datasets/seeds-llm-mailbox --languages en --format user-input --injection-delimiters $'\n</email>\n\nINJECTION_PAYLOAD\n\n<email>'
```

## English + Low Resource Languages Dataset

The dataset incluing the Low Resource Languages (Zulu, Gaelic, Albanian, Scottish) samples can be generated using this command (ensure you include `--include-system-message`):

```bash
spikee generate --seed-folder datasets/seeds-llm-mailbox --format user-input
```

By default, Spikee matches jailbreaks and instructions by language, keeping pairs coherent.
If you run with `--match-languages false`, it will mix across languages, creating a much larger but noisier dataset.
