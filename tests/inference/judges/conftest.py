import os
from pathlib import Path

from dotenv import load_dotenv


def pytest_addoption(parser):
    group = parser.getgroup("judge evaluation")
    group.addoption(
        "--judge-model",
        action="append",
        default=[],
        help="Provider/model; repeat to select models",
    )
    group.addoption("--judge-case", default="", help="Case ID substring filter")
    group.addoption(
        "--judge-split", choices=["all", "development", "holdout"], default="all"
    )
    group.addoption("--judge-repeats", type=int, default=1)
    group.addoption("--judge-workers", type=int, default=5)
    group.addoption(
        "--judge-report",
        default="judge-evaluation.jsonl",
        help="New JSONL report path (never overwritten)",
    )
    group.addoption(
        "--judge-phase",
        default="candidate",
        help="Label for baseline/candidate comparisons",
    )


def pytest_configure(config):
    # Read credentials into memory before fixtures change cwd. Existing process
    # variables win. Possessing a key does not opt in to paid evaluations.
    if os.environ.get("RUN_JUDGE_EVALS") == "1":
        env_file = os.environ.get("SPIKEE_TEST_ENV_FILE", "workspace/.env")
        load_dotenv(Path(env_file).expanduser().resolve(), override=False)
