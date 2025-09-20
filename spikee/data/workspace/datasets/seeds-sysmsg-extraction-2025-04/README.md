# Dataset generation

## English dataset (bare)
The English dataset can be generated using this command (ensure you include `--include-system-message`):

```bash
spikee generate --seed-folder datasets/seeds-sysmsg-extraction-2025-04 --include-system-message --languages en
```

## English dataset (spotlighting)
You can add XML and JSON spotlighting like this:

```bash
spikee generate --seed-folder datasets/seeds-sysmsg-extraction-2025-04 --include-system-message --languages en --spotlighting-data-markers $'\n<data>\nDOCUMENT\n</data>\n',$'\n{"document":"DOCUMENT"}\n'
```

## English + Low Resource Languages Dataset (bare)

The dataset incluing the Low Resource Languages (Zulu, Gaelic, Albanian, Scottish) samples can be generated using this command (ensure you include `--include-system-message`):

```bash
$ spikee generate --seed-folder datasets/seeds-sysmsg-extraction-2025-04 --include-system-message 
```

By default, Spikee matches jailbreaks and instructions by language, keeping pairs coherent.
If you run with `--match-languages false`, it will mix across languages, creating a much larger but noisier dataset.
