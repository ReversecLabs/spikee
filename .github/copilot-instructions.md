# Spikee AI Agent Instructions

Spikee is a modular LLM security testing toolkit (v0.7.3-dev) for prompt injection/jailbreak assessment.

## Core Architecture

**Two-Stage Pipeline**: `generator.py` → `tester.py` → `results.py`
- Generator composes JSONL datasets from seed folders, applying plugin transformations
- Tester executes entries via `AdvancedTargetWrapper` (handles retries, throttling, multi-turn)
- Results analyzer parses JSONL into HTML reports with breakdowns by jailbreak/instruction/plugin

**Module Loading System** (`utilities/modules.py`):
```python
# Resolution order: workspace/{type}/ → spikee/{type}/ → spikee/data/
load_module_from_path(name, module_type):  # Returns class instance OR legacy module
  1. Try os.path.join(cwd, module_type, f"{name}.py")
  2. Fall back to importlib.import_module(f"spikee.{module_type}.{name}")
  3. Call _resolve_impl_class() to find concrete subclass via inspect.getmembers()
  4. Return _instantiate_impl() OR raw module for legacy compatibility
```

**OOP Migration Pattern** (v0.5.0 → v1.0.0 deprecation):
- All new modules inherit from `spikee/templates/{target,plugin,attack,judge}.py` ABC classes
- Legacy function-based modules (`def process_input()` at module level) still load but deprecated
- Code uses `inspect.signature()` to detect parameter support (e.g., `"plugin_option" in params`)
- Use `hasattr(module, "transform")` to check for methods before calling

## Content Type System (`utilities/hinting.py`)

**CRITICAL**: The `Content` type wraps multimodal data flowing through the pipeline:

```python
from spikee.utilities.hinting import Content, Audio, Image, content_factory, get_content, get_content_type

Content = Union[str, Audio, Image]  # str for text, Audio/Image for multimodal

# Create typed content
content_factory(raw_data, content_type="text")   # → str
content_factory(raw_data, content_type="audio")  # → Audio(raw_data)
content_factory(raw_data, content_type="image")  # → Image(raw_data)

# Inspect content
get_content(content)       # → raw str/bytes (unwraps wrapper)
get_content_type(content)  # → "text" | "audio" | "image"

# Validate plugin/target accepts the content type
validate_content_signature(content, function, "param_name")  # → bool
validate_content_annotation(content, annotation)             # → bool
```

**Type Aliases** (for use in signatures):
```python
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint, TargetResponseHint, AttackResponseHint

ModuleDescriptionHint = Tuple[List[ModuleTag], str]
ModuleOptionsHint     = Tuple[List[str], bool]
TargetResponseHint    = Union[Content, bool, Tuple[Union[Content, bool], Any]]
AttackResponseHint    = Tuple[int, bool, Union[Content, Dict[str, Any]], Content]
```

**Dataset entry format change**: Entries now use `"content"` + `"content_type"` fields instead of `"text"`. Legacy `"text"` field is still read by `process_entry()` for backward compatibility.

## Module Base Class Pattern

**CRITICAL**: All templates inherit from `Module` ABC (`templates/module.py`). Every module MUST implement:

```python
from spikee.templates.module import Module
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint

class MyModule(Module):
    def get_description(self) -> ModuleDescriptionHint:
        # MUST return Tuple[List[ModuleTag], str] - used by 'spikee list'
        return [ModuleTag.SINGLE], "Brief description of what this module does"
    
    def get_available_option_values(self) -> ModuleOptionsHint:
        # CRITICAL RETURN TYPE: Tuple[List[str], bool]
        # Missing type hints break isinstance() checks in get_default_option()
        # First item in options_list is the DEFAULT used when no option specified
        # Bool indicates if module needs LLM provider to operate (affects error messages)
        return ["mode=aggressive", "mode=stealth"], False  # No LLM needed
        # OR
        return ["model=gpt-4o", "model=claude-3.5"], True   # Requires LLM
```

**Type Hint Requirements**:
- `get_default_option()` checks `isinstance(available, tuple)` - returns `None` if not tuple
- CLI validates LLM requirement flag before running - prevents cryptic API key errors
- `spikee list` uses return types to format output tables correctly
- **Always import typing types** - runtime type checks depend on proper annotations

## Template Contracts

### Target (`spikee/templates/target.py`)
```python
from typing import Optional
from spikee.utilities.hinting import Content, TargetResponseHint, ModuleDescriptionHint, ModuleOptionsHint

class MyTarget(Target):
    def __init__(self):
        super().__init__()
        # Configure turn support - tester.py checks this via target_module.config
        self.config: Dict[str, bool] = {
            "single-turn": True,   # Accepts string input
            "multi-turn": True,    # Accepts list[str] for conversation
            "backtrack": False     # Supports removing last turn
        }
    
    def process_input(self, input_text: Content, system_message: Optional[Content] = None, 
                     target_options: Optional[str] = None) -> TargetResponseHint:
        # CRITICAL TYPE: Return Content for LLM outputs, bool for guardrails (True = bypassed)
        # AdvancedTargetWrapper relies on isinstance(response, bool) to detect guardrail mode
        # AdvancedTargetWrapper validates content type via validate_content_signature() before calling
        # Raise GuardrailTrigger(msg, categories={}) for blocked payloads
        # Raise RetryableError(msg, retry_period=60) for 429/throttling
        pass
    
    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.SINGLE, ModuleTag.MULTI], "My custom target implementation"
    
    def get_available_option_values(self) -> ModuleOptionsHint:
        # First option is default, bool=True means needs LLM provider
        return ["temperature=0.7", "temperature=0.0"], True
```

**`ProviderTarget` removed** — `llm_provider.LLMProvider` now inherits directly from `Target`. Custom targets that wrapped `ProviderTarget` should migrate to inheriting from `Target` or instantiating `LLMProvider` directly.

### Plugin (`spikee/templates/plugin.py` or `basic_plugin.py`)
```python
from typing import Optional, List, Union
from spikee.utilities.hinting import Content, ModuleDescriptionHint, ModuleOptionsHint

class MyPlugin(BasicPlugin):  # BasicPlugin auto-handles exclude_patterns via regex split
    def plugin_transform(self, text: str, plugin_option: str = "") -> str:
        # Transform only non-excluded chunks (text plugins only)
        return text.upper()
    
    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.SINGLE, ModuleTag.ENCODING], "Uppercase transformation plugin"
    
    def get_available_option_values(self) -> ModuleOptionsHint:
        return ["variants=50", "variants=100"], False  # No LLM needed
        
# OR for full control (including multimodal):
class MyPlugin(Plugin):
    def transform(
        self, content: Content, exclude_patterns: Optional[List[str]] = None, plugin_option: str = ""
    ) -> Union[Content, List[Content]]:
        # CRITICAL: Parameter must be named 'content' for Content types, or 'text' for str-only
        # Generator inspects annotation to check type compatibility before calling
        # Return Content for single variant, List[Content] for N variants
        return [content_transform_a(content), content_transform_b(content)]
```

**Plugin content-type routing**: Generator's `apply_plugin()` uses `validate_content_annotation()` to check if plugin accepts the current content type. It first looks for a `content: Content` parameter, then falls back to `text: str`. Mismatched types raise `ValueError` — annotate parameters correctly.

**Multimodal plugins** (`plugins/tts.py`, `plugins/text2image.py`):
```python
from spikee.utilities.hinting import Audio, Image

class TTSPlugin(Plugin):
    def transform(self, content: str, ...) -> Audio:   # str in → Audio out
        return Audio(base64_bytes)

class Text2ImagePlugin(Plugin):
    def transform(self, content: str, ...) -> Image:   # str in → Image out
        return Image(base64_bytes)
```

### Attack (`spikee/templates/attack.py`)
```python
from spikee.utilities.hinting import Content, AttackResponseHint

class MyAttack(Attack):
    def __init__(self):
        super().__init__(turn_type=Turn.SINGLE)  # or Turn.MULTI
    
    def attack(self, entry: Dict[str, Any], target_module, call_judge: Callable,
               max_iterations: int, attempts_bar=None, bar_lock=None,
               attack_options=None) -> AttackResponseHint:
        # CRITICAL RETURN TYPE: Tuple[int, bool, Union[Content, Dict[str, Any]], Content]
        # attack_options param is NEW - accept it even if unused for forward compatibility
        # Use Attack.standardised_input_return() for dict return with conversation/objective
        for i in range(max_iterations):
            with bar_lock:
                attempts_bar.update(1)
            modified = modify_payload(entry["payload"])
            response, _ = target_module.process_input(modified, entry.get("system_message"))
            if call_judge(entry, response):
                return (i+1, True, modified, response)
        return (max_iterations, False, modified, response)
```

### Judge (`spikee/templates/judge.py`)
```python
from spikee.utilities.hinting import Content

class MyJudge(Judge):
    def judge(self, llm_input: Content, llm_output: Content, judge_args="", 
             judge_options="") -> bool:
        # CRITICAL RETURN TYPE: bool (True = attack succeeded, False = failed)
        # llm_input and llm_output may now be Audio/Image Content objects (multimodal)
        return judge_args in llm_output
```

### Provider (`spikee/templates/provider.py`)

New template base for LLM/TTS/STT providers. Streaming variant in `templates/streaming_provider.py`:
```python
from spikee.templates.streaming_provider import StreamingProvider
from typing import Callable, Sequence, Union
from spikee.utilities.hinting import Content

class MyStreamingProvider(StreamingProvider):
    def invoke_streaming(self, messages, callback: Callable) -> None:
        # Stream tokens via callback
        pass
```

`Provider.default_timeout` property reads `SPIKEE_API_TIMEOUT` env var as a global timeout fallback.

## Exception-Based Control Flow

**Custom Exceptions** (`tester.py`):
```python
GuardrailTrigger(message, categories={})  # Raised by targets when blocked - triggers retry
RetryableError(message, retry_period=60)   # 429/throttling - triggers exponential backoff
MultiTurnSkip(message)                     # Single-turn target got multi-turn input - skip entry
```

**AdvancedTargetWrapper Pattern** (wraps `process_input()` with retry logic):
- Introspects target signature via `inspect.signature()` to detect optional params and type hints
- Only passes kwargs target actually supports (`target_options`, `logprobs`, `spikee_session_id`, etc.)
- **Type hints on optional parameters help introspection** - wrapper checks `param.annotation` to determine types
- Catches exceptions above and retries up to `max_retries`, respecting `throttle` delays
- Returns `(response: Content | bool, meta: Any)` tuple - preserves Content wrapper for judge
- Uses `validate_content_signature()` to confirm content type compatibility before calling target

## Data Flow Internals

**Generator Pipeline** (`generator.py`):
1. `Entry()` class builds dataset entries with `EntryType` enum (now in `utilities/enums.py`)
2. `Entry` uses `content: Content` field (replacing `text`) + `payload: Content`
3. `insert_jailbreak()` injects payloads and preserves content type via `content_factory()`
4. `apply_plugin()` checks annotation via `validate_content_annotation()` to route `content` vs `text` param, handles piping via `~` separator
5. Plugins returning `List[Content]` create N dataset variants per base entry
6. `Entry.to_entry()` serializes to JSONL with `"content"` + `"content_type"` fields (replaces `"text"`)

**Tester Workflow** (`tester.py`):
```python
process_entry(entry, target_module, ...):
  1. Reads "content"+"content_type" from entry (falls back to "text" for legacy datasets)
  2. _do_single_request() - standard attempts (up to --attempts)
  3. If all fail AND attack_module exists:
     - Call attack_module.attack() with wrapped target (passes attack_options kwarg)
     - Attack updates attempts_bar with bar_lock per iteration
     - Returns first success OR final failure
  4. Result dicts include "input_type" and "response_type" fields
  5. Append result dicts to output_file via append_jsonl_entry()
```

**Multi-Turn Support** (`StandardisedConversation` in `templates/`):
- Tree structure: `{msg_id: {"parent": int, "children": [], "data": Any, "attempt": bool}}`
- Root always node 0, tracks conversation branches for backtracking attacks
- Serialized to JSON string in result dict `"conversation"` field

## Signature Introspection Patterns

**Backward Compatibility Check** (appears throughout codebase):
```python
sig = inspect.signature(module.transform)  # or .attack, .judge, .process_input
params = sig.parameters

# Check parameter existence (type hints optional but recommended)
if "plugin_option" in params:
    result = module.transform(text, exclude_patterns, plugin_option)
else:
    result = module.transform(text, exclude_patterns)  # Legacy module

# Type hint checking for advanced behavior
if "plugin_option" in params:
    param = params["plugin_option"]
    # param.annotation available if type hints present: Optional[str], str | None, etc.
    if param.default is inspect.Parameter.empty:
        # Required parameter - always pass
        pass
```

**Module Instance vs Legacy Module**:
- `_resolve_impl_class()` uses `inspect.getmembers()` to find concrete subclass
- If found: instantiate and return class instance
- If not found: return raw module for legacy function-based hooks
- `get_options_from_module()` tries `module.get_available_option_values()` first, then instantiates if `inspect.ismodule()`
- **Type checking pattern**: `isinstance(available, tuple)` used to detect proper return types
- Methods without type hints still work but may fail `isinstance()` checks in some code paths

**Plugin `exclude_patterns` Handling**:
- `BasicPlugin` base class auto-implements via `re.split(compound_regex, text)`
- Chunks matching `exclude_patterns` preserved verbatim, others transformed
- Pattern: `compound = "(" + "|".join(exclude_patterns) + ")"` then `re.fullmatch(chunk)`
- `BasicPlugin.transform()` parameter is named `content` (not `text`) for routing consistency

**Progress Bar Management**:
- All modifications require `with bar_lock: attempts_bar.update(1)`
- On early success, adjust `attempts_bar.total` to skip remaining iterations:
  ```python
  attempts_bar.total = attempts_bar.total - (planned_iterations - actual_iterations)
  ```

**Dataset Entry Long IDs**:
- Format: `{task_type}_{doc_id}_{jailbreak_id}_{instruction_id}_{position}{plugin_suffix}`
- Suffixes: `-p{prefix_id}`, `-s{suffix_id}`, `-sys` (system message), `{attack_name}` (dynamic)
- Used for result grouping and resume file matching

**Resume Logic** (`tester.py` regions):
- `_find_resume_candidates()` uses `build_resource_name()` for exact tag matching
- Interactive prompt via `InquirerPy` if TTY detected
- `--auto-resume` silently picks latest, `--no-auto-resume` forces fresh

## Testing Patterns

**Functional Tests** (`tests/functional/`):
- Use `tmp_path` fixture + `subprocess.run(["spikee", "generate", ...])` 
- Helpers in `utils.py`: `_split_base_and_plugin_entries()`, `_load_plugin_module()`
- Tests verify JSONL structure, plugin transformations, resume merging

**Inference Tests** (`tests/inference/`):
- Parameterized via `@pytest.mark.parametrize` over target/attack/judge combinations
- Require API keys in `.env` - skip if missing

## Type Hinting Philosophy

**Critical Type Hints** (breaks functionality if missing):
1. **Module methods**: `get_description() -> ModuleDescriptionHint` and `get_available_option_values() -> ModuleOptionsHint`
   - Import from `spikee.utilities.hinting` — these are aliases for `Tuple[List[ModuleTag], str]` / `Tuple[List[str], bool]`
   - `get_default_option()` checks `isinstance(available, tuple)` - returns `None` on type mismatch
   - Missing tuple type breaks default option resolution and CLI startup

2. **Attack return types**: `attack(...) -> AttackResponseHint`
   - Expands to `Tuple[int, bool, Union[Content, Dict[str, Any]], Content]`
   - Tester unpacks: `attempts, success, modified, response = attack_module.attack(...)`
   - Wrong tuple size or element types cause immediate crashes

3. **Plugin return variance**: `transform(...) -> Content | List[Content]`
   - Generator checks `isinstance(result, list)` to detect multi-variant behavior
   - Plugin parameter must be named `content` (annotated `Content`) or `text` (annotated `str`)
   - Mismatched annotations raise `ValueError` - annotate parameters correctly

4. **Target return discrimination**: `process_input(...) -> TargetResponseHint`
   - `bool` return detected via `isinstance(response, bool)` for guardrail mode
   - `Content` return carries actual LLM output (may be Audio/Image for multimodal)
   - Type confusion breaks success detection

**Recommended Type Hints** (improves introspection):
- Optional parameters: Use `Optional[str]` or `str | None` for signature introspection
- Callables: Type `call_judge: Callable` for better IDE support and documentation
- Dicts: Specify `Dict[str, Any]` for `entry` parameters to clarify structure
- Config dicts: Type `self.config: Dict[str, bool]` to document expected keys

**Import Pattern**:
```python
from typing import Tuple, List, Optional, Dict, Any, Callable, Union
from spikee.utilities.hinting import Content, Audio, Image, ModuleDescriptionHint, ModuleOptionsHint, TargetResponseHint, AttackResponseHint
```

## Common Code Patterns

**Options String Parsing**:
```python
# Input: "plugin1:key=val,key2=val2;plugin2:key=val"
parse_plugin_options(str) -> Dict[str, str]  # Returns dict[plugin_name, option_string]
parse_options(str) -> Dict[str, str]         # Single module's options as dict[key, value]
```

**JSON Extraction from LLM** (`utilities/modules.py`):
- `extract_json_or_fail()` - strips markdown fences, fixes unescaped quotes, balanced-bracket scan
- Used by LLM-based plugins/attacks that expect structured output

**Guardrail Testing**:
- Target returns `bool`: True = bypassed (success), False = blocked (failure)
- OR raises `GuardrailTrigger` with optional `categories` dict for categorization

## Development Workflow

```bash
pip install -e .                 # Editable install - no reinstall needed
pytest tests/functional -v       # Run integration tests (no API keys)
pytest tests/inference -v        # Run LLM tests (requires .env)
spikee list {targets|plugins}    # Discover available modules
```

## Commit Prefixes

`feat:`, `fix:`, `change:`, `dataset:`, `dev:`, `docs:` - only first 4 in CHANGELOG.md