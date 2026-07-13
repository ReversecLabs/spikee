import pytest

from spikee.utilities.modules import load_module_from_path
from spikee.utilities.files import read_jsonl_file
from spikee.templates.judge import Judge
from ..utils import spikee_test_cli


def test_judge_canary():
    judge = load_module_from_path("canary", "judges")

    assert isinstance(judge, Judge), (
        f"Canary judge should be an instance of Judge: {type(judge)}"
    )

    canary = "CANARY_STRING"
    test_cases = {
        "This response contains the CANARY_STRING.": True,
        "This response does not have it.": False,
    }

    for response, expected in test_cases.items():
        result = judge.judge("", response, canary)

        assert result == expected, (
            f"Expected {expected} for response: '{response}', got {result}"
        )


def test_llm_judge_regex():
    judge = load_module_from_path("regex", "judges")

    assert isinstance(judge, Judge), (
        f"Regex judge should be an instance of Judge: {type(judge)}"
    )

    test_cases = {
        "The answer is 42.": True,
        "No numbers here!": False,
        "Multiple numbers: 1, 2, 3.": True,
    }

    regex = r"\d+"  # Matches any string containing a digit

    for response, expected in test_cases.items():
        result = judge.judge("", response, regex)

        assert result == expected, (
            f"Expected {expected} for response: '{response}', got {result}"
        )


def test_judge_secret_leak():
    judge = load_module_from_path("secret_leak", "judges")

    assert isinstance(judge, Judge), (
        f"Secret leak judge should be an instance of Judge: {type(judge)}"
    )

    # Positives: one representative per HIGH-SIGNAL default category. The
    # default scan deliberately excludes email/ip (see DEFAULT_CATEGORIES).
    positives = [
        "Exfiltrated credential: AKIAIOSFODNN7EXAMPLE",
        "Token: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N",
        "Use sk-proj-abcdEFGH1234ijklMNOP5678 to authenticate",
        "Anthropic key sk-ant-api03-abcdEFGH1234ijklMNOP5678",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIB...\n-----END RSA PRIVATE KEY-----",
        "Card on file: 4111 1111 1111 1111",  # Luhn-valid test PAN
    ]
    for response in positives:
        assert judge.judge("", response, "") is True, (
            f"Expected a leak to be detected in: {response!r}"
        )

    # Negatives: benign prose and near-misses that must NOT trigger.
    negatives = [
        "The quick brown fox jumps over the lazy dog.",
        "Order number 4111 1111 1111 1112 was rejected.",  # fails Luhn
        "AKIA is the prefix used by AWS access keys.",  # prefix only, no body
        # email/ip are NOT in the default category set: a benign example
        # address or host must not be labelled a successful exfiltration.
        "Contact victim at jane.doe@example.com for details",
        "Internal host reachable at 10.0.12.34",
    ]
    for response in negatives:
        assert judge.judge("", response, "") is False, (
            f"Expected no leak to be detected in: {response!r}"
        )

    # email/ip ARE still detected when explicitly selected (opt-in) or via "all".
    assert judge.judge("", "Contact jane.doe@example.com", "email") is True
    assert judge.judge("", "Host 10.0.12.34", "ip") is True
    assert judge.judge("", "Contact jane.doe@example.com", "all") is True


def test_judge_secret_leak_category_selection():
    judge = load_module_from_path("secret_leak", "judges")

    aws = "Leaked AKIAIOSFODNN7EXAMPLE here"
    email = "Reach me at jane.doe@example.com"

    # Selecting only aws_access_key matches the AWS key but ignores the email.
    assert judge.judge("", aws, "aws_access_key") is True
    assert judge.judge("", email, "aws_access_key") is False

    # judge_args accepts a comma-separated string and a list of categories.
    assert judge.judge("", email, "aws_access_key,email") is True
    assert judge.judge("", email, ["aws_access_key", "email"]) is True

    # judge_options (CLI override) takes precedence over judge_args and
    # tolerates the "judge_name:" prefix form.
    assert judge.judge("", email, "aws_access_key", "secret_leak:email") is True
    assert judge.judge("", aws, "email", "aws_access_key") is True

    # Unknown categories are rejected loudly.
    with pytest.raises(ValueError):
        judge.judge("", aws, "not_a_category")


@pytest.mark.parametrize("judge_variant", ["test_judge", "test_judge_legacy"])
def test_spikee_test_custom_judge_default_mode(
    run_spikee, workspace_dir, judge_variant
):
    """Test custom judges across OOP and legacy implementations.

    Uses always_success target for consistent output - target variation doesn't affect
    judge behavior since both OOP and legacy targets produce identical outputs.
    """
    dataset_filename = (
        "test_judge_dataset_legacy.jsonl"
        if judge_variant.endswith("_legacy")
        else "test_judge_dataset.jsonl"
    )
    dataset_path = workspace_dir / "datasets" / dataset_filename
    assert dataset_path.exists()

    results_file, _ = spikee_test_cli(
        run_spikee,
        workspace_dir,
        target="always_success",
        datasets=[dataset_path],
    )

    results = read_jsonl_file(results_file[0])
    assert results
    assert all(not entry["success"] for entry in results)


@pytest.mark.parametrize("judge_variant", ["test_judge", "test_judge_legacy"])
def test_spikee_test_custom_judge_with_options(
    run_spikee, workspace_dir, judge_variant
):
    """Test custom judges with --judge-options across OOP and legacy implementations.

    Uses always_success target for consistent output - target variation doesn't affect
    judge behavior since both OOP and legacy targets produce identical outputs.
    """
    dataset_filename = (
        "test_judge_dataset_legacy.jsonl"
        if judge_variant.endswith("_legacy")
        else "test_judge_dataset.jsonl"
    )
    dataset_path = workspace_dir / "datasets" / dataset_filename
    assert dataset_path.exists()

    results_file, _ = spikee_test_cli(
        run_spikee,
        workspace_dir,
        target="always_success",
        datasets=[dataset_path],
        additional_args=[
            "--judge-options",
            f"{judge_variant}:mode=success",
        ],
    )

    results = read_jsonl_file(results_file[0])
    assert results
    assert all(entry["success"] for entry in results)
