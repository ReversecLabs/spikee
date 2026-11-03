import os
import sys
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if os.environ.get("RUN_INFERENCE_TESTS") == "1":
    load_dotenv(
        Path(os.environ.get("SPIKEE_TEST_ENV_FILE", "workspace/.env"))
        .expanduser()
        .resolve(),
        override=False,
    )

pytest_plugins = ["tests.functional.conftest"]


# The free Groq tier has a small TPM budget. Space test cases without reducing
# their two-call coverage or treating rate limits as successful inference.
_last_groq_test_finished = None


@pytest.fixture(autouse=True)
def pace_groq_tests(request):
    global _last_groq_test_finished
    params = getattr(getattr(request.node, "callspec", None), "params", {})
    uses_groq = any("groq/" in str(value) for value in params.values())
    if not uses_groq or os.environ.get("RUN_INFERENCE_TESTS") != "1":
        yield
        return
    interval = float(os.environ.get("SPIKEE_TEST_GROQ_INTERVAL", "65"))
    if _last_groq_test_finished is not None:
        remaining = interval - (time.monotonic() - _last_groq_test_finished)
        if remaining > 0:
            time.sleep(remaining)
    yield
    _last_groq_test_finished = time.monotonic()
