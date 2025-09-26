# Dataset generation

## English dataset (bare)
The English dataset can be generated using this command:

```bash
spikee generate --seed-folder datasets/seeds-cybersec-2025-04 --languages en
```

## English dataset (spotlighting + system message)
You can add XML/JSON spotlighting and include a system message like this:

```bash
spikee generate --seed-folder datasets/seeds-cybersec-2025-04 --languages en --include-system-message --spotlighting-data-markers $'\n<data>\nDOCUMENT\n</data>\n',$'\n{"document":"DOCUMENT"}\n'
```

## English + Low Resource Languages Dataset (bare)

The dataset including Low Resource Languages (Zulu, Gaelic, Albanian, Scottish) can be generated with:

```bash
spikee generate --seed-folder datasets/seeds-cybersec-2025-04
```

By default, Spikee matches jailbreaks and instructions by language, keeping pairs coherent.
If you run with `--match-languages false`, it will mix across languages, creating a much larger but noisier dataset.

