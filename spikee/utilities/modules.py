import importlib
import inspect
import os
import re
import json
from typing import Any, Dict, Optional


def _resolve_impl_class(module, module_type):
    """Return the first concrete implementation class for the given module."""

    match module_type:
        case "targets":
            from spikee.templates.target import Target

            base_classes = (Target,)

        case "judges":
            from spikee.templates.judge import Judge
            from spikee.templates.llm_judge import LLMJudge

            base_classes = (Judge, LLMJudge)

        case "plugins":
            from spikee.templates.plugin import Plugin

            base_classes = (Plugin,)

        case "attacks":
            from spikee.templates.attack import Attack

            base_classes = (Attack,)

        case "providers":
            from spikee.templates.provider import Provider

            base_classes = (Provider,)

        case _:
            raise ValueError(
                f"Unknown module type '{module_type}' for implementation resolution"
            )

    if not base_classes or not inspect.ismodule(module):
        return None

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj in base_classes:
            continue
        if obj.__module__ != module.__name__:
            continue
        if issubclass(obj, base_classes):
            return obj
    return None


def _instantiate_impl(module, module_type):
    impl_class = _resolve_impl_class(module, module_type)
    return impl_class() if impl_class else None


# ==== Loading Modules ====
def load_module_from_path(name, module_type):
    """Loads a module either from a local path or from the spikee package."""
    try:
        local_path = os.path.join(os.getcwd(), module_type, f"{name}.py")
        if os.path.isfile(local_path):
            spec = importlib.util.spec_from_file_location(name, local_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            else:
                raise ImportError(f"Could not load module {name} from {local_path}")
        else:
            mod = importlib.import_module(f"spikee.{module_type}.{name}")

    except ModuleNotFoundError as e:
        trimmed = str(e).split("No module named ")[-1].strip("'\"")

        if trimmed == name or trimmed.endswith(f".{name}"):
            raise ImportError(
                f"[Import Error] Module '{name}' not found locally or built-in. Use 'spikee list {module_type}' to see available options."
            )

        else:
            raise ImportError(
                f"[Import Error] Module {name}, dependency '{trimmed}' not found - review {name} and ensure all required dependencies are installed."
            )

    instance = _instantiate_impl(mod, module_type)
    if instance is not None:
        return instance

    return mod


def get_options_from_module(module, module_type=None):
    """
    Return the option values advertised by the given module or instance.

    Args:
        module: Either an instantiated module (new OOP) or the imported module.
        module_type: Optional str specifying the module category. Required when
            `module` is a module rather than an instance.
    """
    if module and hasattr(module, "get_available_option_values"):
        return module.get_available_option_values()

    if inspect.ismodule(module) and module_type:
        instance = _instantiate_impl(module, module_type)
        if instance and hasattr(instance, "get_available_option_values"):
            return instance.get_available_option_values()

    return None


def get_description_from_module(module, module_type=None):
    """
    Return the description advertised by the given module or instance.

    Args:
        module: Either an instantiated module (new OOP) or the imported module.
        module_type: Optional str specifying the module category. Required when
            `module` is a module rather than an instance.
    """
    if module and hasattr(module, "get_description"):
        return module.get_description()

    if inspect.ismodule(module) and module_type:
        instance = _instantiate_impl(module, module_type)
        if instance and hasattr(instance, "get_description"):
            return instance.get_description()

    return None


def get_default_option(module, module_type=None):
    available = get_options_from_module(module, module_type)

    if isinstance(available, tuple):
        available = available[0]

    return available[0] if available else None


def parse_options(option: Optional[str]) -> Dict[str, str]:
    opts: Dict[str, str] = {}
    if not option:
        return opts
    for p in (x.strip() for x in option.split(",") if x.strip()):
        if "=" in p:
            k, v = p.split("=", 1)
            opts[k.strip()] = v.strip()
    return opts


def extract_json_or_fail(text: str) -> Dict[str, Any]:
    """
    Robust JSON extractor.
    """
    if not text:
        raise RuntimeError("LLM returned empty response")

    t = text.strip()

    # 1) fenced code block
    m = re.search(r"```(?:json)?\s*(.*?)```", t, flags=re.IGNORECASE | re.DOTALL)
    if m:
        t = m.group(1).strip()

    # 2) try direct JSON parse
    try:
        return json.loads(t)
    except Exception:
        pass

    # 3) fix unescaped quotes and try again
    t_fixed = fix_unescaped_quotes(t)
    try:
        return json.loads(t_fixed)
    except Exception:
        pass

    # 4) scan for first balanced {...}
    start = -1
    depth = 0
    for i, ch in enumerate(t):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidate = t[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        start = -1

    raise RuntimeError(f"LLM did not return valid JSON object: \n\n {text}")


def fix_unescaped_quotes(text: str) -> str:
    """
    Fix unescaped quotes within JSON string values by properly escaping them.

    Args:
        text (str): The input text with potentially unescaped quotes in JSON

    Returns:
        str: Text with properly escaped quotes for valid JSON
    """
    result = []
    i = 0
    STATE_NORMAL = 0  # Outside any string
    STATE_IN_STRING = 1  # Inside a JSON string value

    state = STATE_NORMAL

    while i < len(text):
        char = text[i]

        if state == STATE_NORMAL:
            result.append(char)
            if char == '"':
                state = STATE_IN_STRING

        elif state == STATE_IN_STRING:
            # If we have an escape character, add it and the next character
            if char == "\\":
                result.append(char)
                i += 1
                if i < len(text):
                    result.append(text[i])
            # If we have an unescaped quote, check if it's closing the string or internal
            elif char == '"':
                # Look ahead to determine if this is likely closing the string value
                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1

                # If followed by these characters, it's likely closing the string
                if j < len(text) and (
                    text[j] == "," or text[j] == "}" or text[j] == ":" or text[j] == "]"
                ):
                    result.append(char)  # Closing quote, don't escape
                    state = STATE_NORMAL
                else:
                    # Internal quote, needs escaping
                    result.append("\\")
                    result.append(char)
            else:
                result.append(char)

        i += 1

    return "".join(result)
