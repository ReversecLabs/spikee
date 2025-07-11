# Developing Custom Plugins

Plugins are Python scripts that transform payloads during dataset generation (`spikee generate`). They are used to create variations of your test cases *before* testing begins.

### Plugins vs. Dynamic Attacks: What's the Difference?

Both Plugins and Dynamic Attacks can generate variations of a payload, but they serve different purposes in the testing workflow:

*   **Plugins (Pre-Test Transformation):**
    *   **When they run:** During `spikee generate`.
    *   **What they do:** Create multiple variations of a payload. Each variation is saved as a **separate, independent entry** in the final dataset file.
    *   **Result:** When you run `spikee test`, every single variation generated by the plugin is tested against the target. This is useful for systematically evaluating a target's resilience to a known set of transformations (e.g., "Is the target vulnerable to Base64 encoding? To Leetspeak?").

*   **Dynamic Attacks (Real-Time Transformation):**
    *   **When they run:** During `spikee test`, but *only if* the initial, standard prompt fails.
    *   **What they do:** Generate and test variations one by one in real-time. The attack stops as soon as a variation succeeds.
    *   **Result:** Only the first successful variation (or the final failed attempt) is logged. This is useful for efficiently finding *any* successful bypass, rather than testing every possible variation.

In short, use **Plugins** to build a comprehensive dataset of known transformations. Use **Dynamic Attacks** to find a single successful bypass with adaptive, real-time logic.

## Plugin Structure

Every plugin is a Python module located in the `plugins/` directory of your workspace. Spikee identifies plugins by their filename.

*   **Location:** `./plugins/my_leetspeak_plugin.py`
*   **Required Function:** `transform`

## The `transform` Function

This is the core function of every plugin. It receives a payload string and returns one or more transformed versions.

### Basic Signature
```python
from typing import List, Union

def transform(text: str, exclude_patterns: List[str] = None) -> Union[str, List[str]]:
    """
    Transforms the input payload text.
    """
    # Your transformation logic here...
```

### Signature with Options Support
For more advanced plugins, you can accept a configuration string and advertise the available options.
```python
from typing import List, Union

def get_available_option_values() -> List[str]:
    """Returns a list of supported option strings for this plugin."""
    return ["mode=strict", "mode=full"] # "mode=strict" is the default

def transform(text: str, exclude_patterns: List[str] = None, plugin_option: str = None) -> Union[str, List[str]]:
    """Transforms the payload based on the provided option."""
    # Your transformation logic here...
```

### Parameters

*   `text: str`
    The input payload, which is typically a combination of a jailbreak and a malicious instruction.

*   `exclude_patterns: List[str]`
    A list of regular expression patterns. Your plugin **must not** transform any part of the `text` that matches one of these patterns. This is critical for preserving sensitive parts of a prompt, like URLs or specific keywords.

*   `plugin_option: Optional[str]`
    A string passed from the command line via `--plugin-options` (e.g., `"my_plugin:mode=full,variants=10"`). If your plugin doesn't need configuration, you can omit this parameter.

### Return Values

*   `str`: Return a single transformed string. Spikee will create one new test case from this.
*   `List[str]`: Return a list of transformed strings. Spikee will create a separate test case for **each string in the list**, allowing you to test multiple variations at once.

## Handling Exclude Patterns

Correctly handling `exclude_patterns` is the most important part of writing a robust plugin. You must leave the excluded parts of the string completely untouched. The recommended way to do this is with `re.split`.

```python
import re

def transform(text: str, exclude_patterns: List[str] = None) -> str:
    if not exclude_patterns:
        # No exclusions, transform the whole text
        return apply_my_transformation(text)

    # 1. Create a single regex pattern that captures any of the exclude patterns.
    # The parentheses around the pattern are crucial for re.split to keep the delimiters.
    combined_pattern = "(" + "|".join(exclude_patterns) + ")"
    
    # 2. Split the text by the combined pattern.
    # even-indexed chunks are normal text; odd-indexed chunks are the exclusions.
    chunks = re.split(combined_pattern, text)
    
    # 3. Transform only the non-excluded chunks.
    transformed_chunks = []
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            # This is normal text, apply the transformation
            transformed_chunks.append(apply_my_transformation(chunk))
        else:
            # This is an excluded part, keep it as is
            transformed_chunks.append(chunk)
            
    # 4. Rejoin the chunks into a single string.
    return "".join(transformed_chunks)

def apply_my_transformation(s: str) -> str:
    # Example transformation: convert to uppercase
    return s.upper()
```

## Complete Example: A Configurable Leetspeak Plugin

This plugin converts text to Leetspeak and can generate multiple random variations. It also correctly handles exclusions.

```python
# ./plugins/1337.py
import re
import random
from typing import List, Union

LEET_MAP = {'a': '4', 'e': '3', 'g': '6', 'i': '1', 'o': '0', 's': '5', 't': '7'}
DEFAULT_VARIANTS = 5

# --- Optional: Support for --plugin-options ---
def get_available_option_values() -> List[str]:
    """Advertises supported options for this plugin."""
    return [
        f"variants={DEFAULT_VARIANTS}", # Default
        "variants=N (where N is the number of variations to generate)"
    ]

def _parse_variants(plugin_option: str) -> int:
    if plugin_option and "variants=" in plugin_option:
        try:
            return int(plugin_option.split("=")[1])
        except (ValueError, IndexError):
            pass # Fallback to default if parsing fails
    return DEFAULT_VARIANTS

# --- Main Transformation Logic ---
def _leet_transform(text: str) -> str:
    """Applies a simple, randomized Leetspeak transformation."""
    return "".join(LEET_MAP.get(c.lower(), c) if random.random() > 0.5 else c for c in text)

def transform(text: str, exclude_patterns: List[str] = None, plugin_option: str = None) -> Union[str, List[str]]:
    """
    Generates Leetspeak variations of the input text, respecting exclusions.
    """
    num_variants = _parse_variants(plugin_option)

    # Handle a single variant case without a loop for efficiency
    if num_variants == 1:
        if not exclude_patterns:
            return _leet_transform(text)
        
        # Handle exclusions for a single variant
        pattern = "(" + "|".join(exclude_patterns) + ")"
        chunks = re.split(pattern, text)
        transformed_chunks = [_leet_transform(chunk) if i % 2 == 0 else chunk for i, chunk in enumerate(chunks)]
        return "".join(transformed_chunks)

    # Handle multiple variants
    variations = []
    for _ in range(num_variants):
        if not exclude_patterns:
            variations.append(_leet_transform(text))
            continue
        
        # Handle exclusions for each variant
        pattern = "(" + "|".join(exclude_patterns) + ")"
        chunks = re.split(pattern, text)
        transformed_chunks = [_leet_transform(chunk) if i % 2 == 0 else chunk for i, chunk in enumerate(chunks)]
        variations.append("".join(transformed_chunks))
        
    return variations
```