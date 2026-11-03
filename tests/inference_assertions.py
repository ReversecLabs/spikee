"""Evidence checks shared by live inference tests and offline regressions."""

import re


def assert_attack_inference(data, stdout=""):
    """Require an executed attack or an explicit model refusal, never a hidden error."""
    errors = [str(data["error"])] if data.get("error") else []
    response = str(data.get("response", ""))
    if response.startswith(
        ("Error", "A generic error occurred", "Attacker LLM refused")
    ):
        errors.append(response)
    for attempt in data.get("attempt_history") or []:
        if attempt.get("error"):
            errors.append(str(attempt["error"]))

    # Format errors alone are not evidence of a safety refusal. Inspect only
    # logged model replies, not prompts containing examples of refusal language.
    replies = re.findall(
        r"^\[[^\]\n]+ DEBUG\] Response: (.*?)(?=^\[|\Z)",
        stdout,
        re.MULTILINE | re.DOTALL,
    )
    if "Error generating LLM variants:" in stdout:
        errors.append("LLM variant generation failed and used a fallback")

    if not errors:
        assert any(reply.strip() for reply in replies), (
            "No live model reply was recorded"
        )
        assert data.get("input"), "Attack produced no input for the target"
        assert response, "Attack produced no target response"
        return "target_response"

    refusal = any(
        re.search(
            r"\b(?:cannot|can't|won't|unable to) (?:help|assist|comply|provide)\b",
            reply,
            re.IGNORECASE,
        )
        for reply in replies
    )
    only_refusal_errors = all(
        "invalid response after" in error.lower()
        or "refused to answer" in error.lower()
        for error in errors
    )
    assert only_refusal_errors and refusal, "Attack inference failed: " + "\n".join(
        errors
    )

    return "refusal"
