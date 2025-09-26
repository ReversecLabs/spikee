## Fetch & Convert the WildGuard Harmful Prompts Dataset

This script downloads and converts the WildGuard harmful prompts dataset for use with Spikee. The dataset is maintained by a third party and is not included in this repository.

### Prerequisites

- A valid **Hugging Face token** with access to the WildGuard repo
- **Create a `.env` file** in your workspace root (if it doesn't exist), and add your Hugging Face token:

   ```
   HF_TOKEN=<your_huggingface_token>
   ```

### Fetch & Convert

Run the fetch-and-convert script from the root of your Spikee workspace:

```bash
python datasets/seeds-wildguardmix-harmful/fetch_and_convert_dataset.py
```

- The script will load environment variables from `.env` automatically.
- Make sure `HF_TOKEN` is set; otherwise, the download will fail.

After completion, the converted prompts will be available at:

```
datasets/seeds-wildguardmix-harmful/standalone_user_inputs.jsonl
```

### Generate the Spikee Test Dataset

Use the `spikee` CLI to generate a test dataset for Spikee's tester script:

```bash
spikee generate \
  --seed-folder datasets/seeds-wildguardmix-harmful \
  --include-standalone-inputs
```

### Optional: Plugins & Variations

If you want to apply additional transformations (e.g., character substitutions, obfuscations), use plugins:

```bash
spikee generate \
  --seed-folder datasets/seeds-wildguardmix-harmful \
  --include-standalone-inputs \
  --plugins 1337
```

### Troubleshooting

- **Token errors**: Ensure `HF_TOKEN` is correctly set in your environment.
- **Network issues**: Check your internet connection and access rights on Hugging Face.