"""Live smoke-test models; override IDs without editing the test matrix."""

import os


def configured_model(provider, default):
    name = os.environ.get(f"SPIKEE_TEST_{provider.upper()}_MODEL", default)
    return f"{provider}/{name}"


OPENAI = configured_model("openai", "gpt-5.4-mini")
AZURE = (
    configured_model("azure_openai", "")
    if os.environ.get("SPIKEE_TEST_AZURE_OPENAI_MODEL")
    else None
)
GOOGLE = configured_model("google", "gemini-3.8-flash")
OPENROUTER = configured_model("openrouter", "qwen/qwen3.8-flash")
GROQ = configured_model("groq", "qwen/qwen3.8-27b")
DEEPSEEK = configured_model("deepseek", "deepseek-flash")
TOGETHER = configured_model("togetherai", "MiniMaxAI/MiniMax-M3.1-Flash")
BEDROCK = configured_model("bedrock", "global.anthropic.claude-haiku-5-5")

PROVIDER_CASES = [
    (OPENAI, ["OPENAI_API_KEY"], True),
    (AZURE, ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"], True),
    (GOOGLE, ["GOOGLE_API_KEY"], True),
    (OPENROUTER, ["OPENROUTER_API_KEY"], True),
    (GROQ, ["GROQ_API_KEY"], True),
    (DEEPSEEK, ["DEEPSEEK_API_KEY"], True),
    (TOGETHER, ["TOGETHER_API_KEY"], True),
    (
        BEDROCK,
        ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"],
        True,
    ),
]

PROVIDER_CASES = [case for case in PROVIDER_CASES if case[0] is not None]
