"""Helpers to ensure tiktoken can work offline by downloading the
public mapping file and placing it under the sha1 filename tiktoken
expects.

This module downloads the file at :pydata:`BLOB_URL`, saves it to a
cache location (or a caller-specified path), and sets
``TIKTOKEN_CACHE_DIR`` for the current process only.
"""
import hashlib
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

# The URL whose sha1 is used as the cache filename by tiktoken_ext.openai_public
BLOB_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"


def _sha1_hex(s: str) -> str:
  return hashlib.sha1(s.encode("utf-8")).hexdigest()


def ensure_tiktoken_offline(cache_root: Path | None = None, save_path: Path | None = None) -> Path:
  """Ensure the public tiktoken mapping file is available for offline use.

  Behavior:
  - Downloads the file at ``BLOB_URL`` and writes it to ``save_path`` if
    provided, otherwise to ``cache_root / <sha1(blob_url)>``.
  - Sets ``os.environ['TIKTOKEN_CACHE_DIR']`` for the current process only to
    the directory containing the saved file.
  - Returns the cache root Path (the directory where the file was written).

  Parameters
  - cache_root: Optional base cache directory to use when ``save_path`` is
    not provided. Defaults to ``~/.cache/spikee/tiktoken``.
  - save_path: Optional exact file path or directory to save the downloaded
    mapping. If a directory is provided, the file will be written there using
    the sha1 filename that tiktoken expects.
  """
  cache_key = _sha1_hex(BLOB_URL)

  if cache_root is None:
    cache_root = Path.home() / ".cache" / "spikee" / "tiktoken"

  cache_root = Path(cache_root)
  cache_root.mkdir(parents=True, exist_ok=True)

  # Determine the final target path. If save_path is provided it may be a
  # directory (in which case we place the file named by the sha1 inside it)
  # or a file path.
  if save_path is not None:
    save_path = Path(save_path)
    if save_path.exists() and save_path.is_dir():
      target_path = save_path / cache_key
    else:
      # Ensure parent dir exists
      save_path.parent.mkdir(parents=True, exist_ok=True)
      target_path = save_path
  else:
    target_path = cache_root / cache_key

  if not target_path.exists():
    # Download the mapping from the public blob URL and write to target.
    try:
      with urllib.request.urlopen(BLOB_URL) as src, open(target_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    except urllib.error.HTTPError as e:
      raise RuntimeError(
        f"Failed to download tiktoken file from {BLOB_URL}: {e}") from e

  # The cache dir tiktoken expects is the directory containing the file.
  cache_dir = target_path.parent
  os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)

  return cache_dir
