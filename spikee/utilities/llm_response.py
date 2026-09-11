"""Bounded format repair and diagnostics for structured LLM responses."""

import json
import re
from collections.abc import Callable, Mapping
from typing import TypeVar

from spikee.utilities.llm_message import AIMessage, HumanMessage

T = TypeVar("T")


class LLMResponseError(RuntimeError):
    def __init__(self, message, response, partial=None):
        super().__init__(message)
        self.response = response
        self.partial = partial


class JSONLResponseError(ValueError):
    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial


def unwrap_json_response(text):
    """Remove explicit reasoning prefixes and outer fences, preserving JSON values."""
    if not isinstance(text, str):
        raise ValueError("Expected a text response containing JSON")  # noqa: TRY004 - invalid external response
    text = text.strip()
    # Some gateways return reasoning in content, even omitting the opening tag.
    # Only consider a prefix: a marker inside JSON or a code fence is literal data.
    if not text.startswith(("{", "[", '"', "```")):
        reasoning, marker, answer = text.partition("</think>")
        if marker:
            text = answer.strip()
        elif reasoning.startswith("<think>"):
            raise ValueError("Unterminated reasoning block before JSON response")
    match = re.fullmatch(
        r"```(?:jsonl?|jsonlines)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else text


def parse_json_object(text, required_keys=(), string_keys=()):
    """Accept fences/prose around one object; never invent or rewrite its values."""
    text = unwrap_json_response(text)
    # LLMs sometimes leave literal newlines/tabs inside quoted strings. Preserve
    # those characters as data; incomplete strings/objects still fail decoding.
    try:
        obj = json.loads(text, strict=False)
    except json.JSONDecodeError as original_error:
        # Decode one object, respecting quoted braces and escaped quotes. Do not
        # salvage an inner object from a malformed array or quoted JSON string.
        start = text.find("{")
        if start < 0 or text.startswith(("[", '"')):
            raise
        obj, end = json.JSONDecoder(strict=False).raw_decode(text, start)
        if "{" in text[end:] or "[" in text[end:]:
            raise ValueError(
                "Expected one JSON object, received multiple values"
            ) from original_error
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object")  # noqa: TRY004 - invalid external response
    missing = [key for key in required_keys if key not in obj]
    if missing:
        raise ValueError(f"Missing required JSON fields: {', '.join(missing)}")
    for key in string_keys:
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            raise ValueError(f"Field '{key}' must be a nonempty string")
    return obj


def parse_jsonl_variations(text):
    """Keep valid variations available if a subsequent format repair fails."""
    variations = []
    errors = []
    for number, line in enumerate(unwrap_json_response(text).splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if (
                not isinstance(obj, dict)
                or not isinstance(obj.get("variation"), str)
                or not obj["variation"].strip()
            ):
                raise ValueError(
                    "Expected an object with a nonempty 'variation' string"
                )
            variations.append(obj["variation"])
        except ValueError as exc:
            errors.append(f"Line {number}: {exc}")
    if errors or not variations:
        raise JSONLResponseError(
            errors[0] if errors else "No JSONL variations returned", variations
        )
    return variations


def log_response_failure(llm, reply, context, error, attempts_remaining):
    """Allowlist metadata; never dump request headers or the raw provider object."""

    def field(obj, name):
        return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)

    raw = field(reply, "original_response")
    choices = field(raw, "choices")
    first_choice = (
        choices[0] if isinstance(choices, (list, tuple)) and choices else None
    )
    usage = field(raw, "usage")
    details = {
        "model": field(raw, "model") or field(llm, "model"),
        "requested_max_tokens": field(llm, "max_tokens"),
        "finish_reason": field(first_choice, "finish_reason"),
    }
    details = {
        key: value if type(value) in (str, int) else None
        for key, value in details.items()
    }
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = field(usage, name)
        details[name] = value if type(value) is int else None
    reasoning_tokens = field(
        field(usage, "completion_tokens_details"), "reasoning_tokens"
    )
    if type(reasoning_tokens) is int:
        details["reasoning_tokens"] = reasoning_tokens
    details.update(error=str(error), attempts_remaining=attempts_remaining)
    print(f"[Warning] {context} invalid LLM response; metadata: {json.dumps(details)}")


def query_structured_response(
    llm,
    messages,
    parser: Callable[[str], T],
    *,
    context: str,
    max_attempts: int = 2,
    format_hint: str = "one complete JSON object",
    repair_guidance: str = (
        "Correct the formatting and retain the intended values and all valid entries."
    ),
) -> T:
    """Repair silently; log exhausted format failures. Provider errors propagate."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    request = list(messages)
    best_partial = None
    for attempt in range(max_attempts):
        reply = llm.invoke(request)
        try:
            return parser(reply.content)
        except ValueError as exc:
            remaining = max_attempts - attempt - 1
            partial = getattr(exc, "partial", None)
            if partial and (not best_partial or len(partial) > len(best_partial)):
                best_partial = partial
            if not remaining:
                log_response_failure(llm, reply, context, exc, remaining)
                raise LLMResponseError(
                    f"{context}: invalid response after {max_attempts} attempts: {exc}",
                    reply.content,
                    best_partial,
                ) from exc
            # Keep the original request and only the latest failed reply. Caller
            # messages stay untouched; repeated failures do not grow the history.
            request = [
                *messages,
                AIMessage(reply.content if isinstance(reply.content, str) else ""),
                HumanMessage(
                    f"Your previous response failed parsing or field validation: {exc}\n"
                    f"Return {format_hint}, satisfying the original required fields and types. "
                    f"{repair_guidance} "
                    "Return the complete corrected response, not a patch, without commentary or code fences."
                ),
            ]
    raise AssertionError("Unreachable response loop")
