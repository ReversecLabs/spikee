"""Helpers to ensure tiktoken can work offline using a bundled .tiktoken file.

This module copies the bundled `o200k_base.tiktoken` resource into a
user-cache directory under the sha1 filename tiktoken expects, then sets
the TIKTOKEN_CACHE_DIR environment variable for the current process only.

The bundled file should live in `spikee.data.adaptive_rsa/o200k_base.tiktoken`
which is already present in the repository.
"""
import hashlib
import importlib.resources
import os
import shutil
from pathlib import Path

# The URL whose sha1 is used as the cache filename by tiktoken_ext.openai_public
BLOB_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"

# Where the bundled resource lives relative to the top-level package `spikee`.
RESOURCE_PACKAGE = "spikee"
RESOURCE_REL_PATH = ("data", "adaptive_rsa", "o200k_base.tiktoken")


def _sha1_hex(s: str) -> str:
  return hashlib.sha1(s.encode("utf-8")).hexdigest()


def ensure_tiktoken_offline(cache_root: Path | None = None) -> Path:
  """Ensure the bundled tiktoken file is available for offline use.

  - Copies the bundled resource to ``cache_root / <sha1(blob_url)>`` if missing.
  - Sets ``os.environ['TIKTOKEN_CACHE_DIR']`` for the current process only.
  - Returns the cache root Path.

  This only mutates the current Python process environment and does not write
  any shell startup files or system-wide configuration.
  """
  cache_key = _sha1_hex(BLOB_URL)

  if cache_root is None:
    cache_root = Path.home() / ".cache" / "spikee" / "tiktoken"

  cache_root = Path(cache_root)
  cache_root.mkdir(parents=True, exist_ok=True)

  target_path = cache_root / cache_key

  if not target_path.exists():
    # Copy bundled resource into the cache under the expected name.
    try:
      # Use importlib.resources.files so we can address nested data folders
      # even if they are not Python packages (no __init__.py).
      res = importlib.resources.files(RESOURCE_PACKAGE)
      for part in RESOURCE_REL_PATH:
        res = res.joinpath(part)
      with res.open("rb") as src, open(target_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    except FileNotFoundError:
      raise FileNotFoundError(
          f"Bundled tiktoken file {'/'.join(RESOURCE_REL_PATH)} not found in package {RESOURCE_PACKAGE}."
      )

  # Set for current process only (safe; affects only this process and its children)
  os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_root)

  return cache_root
