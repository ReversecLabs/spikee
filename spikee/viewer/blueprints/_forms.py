# spikee/viewer/blueprints/_forms.py
"""Dataclass-based form validation for the generate and test blueprints.

Each form class is a plain dataclass whose ``from_form`` classmethod parses
a Flask ``ImmutableMultiDict``, validates all fields, and raises
``FormValidationError`` on the first problem.  Callers catch that exception
and abort with a 400 response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from werkzeug.datastructures import ImmutableMultiDict


class FormValidationError(ValueError):
    """Raised when a submitted form fails validation."""


# Valid values for the --positions CLI argument (matches cli.py help text).
_VALID_POSITIONS: frozenset[str] = frozenset({"start", "middle", "end"})

# ── Helpers ───────────────────────────────────────────────────────────────────


def _require(value: str, field_name: str) -> str:
    """Return *value* stripped, or raise FormValidationError if empty."""
    stripped = value.strip() if value else ""
    if not stripped:
        raise FormValidationError(f"{field_name} is required.")
    return stripped


def _require_safe_name(value: str, field_name: str) -> str:
    """Return *value* stripped and validated as a plain filename (no traversal).

    Raises FormValidationError if the value is empty or contains path separators
    or ``..`` sequences that could allow path traversal.
    """
    stripped = _require(value, field_name)
    if ".." in stripped or "/" in stripped or "\\" in stripped:
        raise FormValidationError(
            f"{field_name} contains invalid characters ('..', '/', or '\\\\')."
        )
    return stripped


def _optional_str(value: str) -> str:
    """Return *value* stripped, or empty string."""
    return value.strip() if value else ""


def _parse_positive_int(
    value: str, field_name: str, default: int | None = None
) -> int | None:
    """Parse *value* as a positive integer.

    Returns *default* if *value* is empty.  Raises FormValidationError for
    non-integer or non-positive values.
    """
    stripped = value.strip() if value else ""
    if not stripped:
        return default
    try:
        v = int(stripped)
    except ValueError:
        raise FormValidationError(f"{field_name} must be a valid integer.")
    if v <= 0:
        raise FormValidationError(f"{field_name} must be a positive integer.")
    return v


def _parse_non_negative_float(
    value: str, field_name: str, default: float = 0.0
) -> float:
    """Parse *value* as a non-negative float.  Returns *default* if empty."""
    stripped = value.strip() if value else ""
    if not stripped:
        return default
    try:
        v = float(stripped)
    except ValueError:
        raise FormValidationError(f"{field_name} must be a valid number.")
    if v < 0:
        raise FormValidationError(f"{field_name} must be non-negative.")
    return v


def _parse_non_negative_float(
    value: str, field_name: str, default: float = 0.0
) -> float:
    """Parse *value* as a non-negative float.  Returns *default* if empty."""
    stripped = value.strip() if value else ""
    if not stripped:
        return default
    try:
        v = float(stripped)
    except ValueError:
        raise FormValidationError(f"{field_name} must be a valid number.")
    if v < 0:
        raise FormValidationError(f"{field_name} must be non-negative.")
    return v


# ── Generate form ─────────────────────────────────────────────────────────────


@dataclass
class GenerateForm:
    """Validated form data for ``POST /generate/run``."""

    seed_folder: str
    positions: list[str]
    injection_delimiters: str
    spotlighting_data_markers: str
    format: str
    include_system_message: bool
    include_standalone_inputs: bool
    plugins: list[str]
    plugin_options: str
    plugin_only: bool
    languages: str
    match_languages: bool
    instruction_filter: str
    jailbreak_filter: str
    include_fixes: str
    threads: int | None
    tag: str

    @classmethod
    def from_form(cls, f: ImmutableMultiDict) -> GenerateForm:
        """Parse and validate a generate form submission.

        Raises:
            FormValidationError: if any required field is missing or invalid.
        """
        seed_folder = _require_safe_name(f.get("seed_folder", ""), "Seed folder")

        positions = f.getlist("positions")
        for pos in positions:
            if pos.strip() not in _VALID_POSITIONS:
                raise FormValidationError(
                    f"Invalid position {pos!r}. Must be one of: {', '.join(sorted(_VALID_POSITIONS))}."
                )

        threads = _parse_positive_int(f.get("threads", ""), "Threads")

        plugin_lines = [
            ln.strip() for ln in f.get("plugins", "").splitlines() if ln.strip()
        ]
        plugin_options = ";".join(
            ln.strip() for ln in f.get("plugin_options", "").splitlines() if ln.strip()
        )

        return cls(
            seed_folder=seed_folder,
            positions=positions,
            injection_delimiters=_optional_str(f.get("injection_delimiters", "")),
            spotlighting_data_markers=_optional_str(
                f.get("spotlighting_data_markers", "")
            ),
            format=_optional_str(f.get("format", "")),
            include_system_message=bool(f.get("include_system_message")),
            include_standalone_inputs=bool(f.get("include_standalone_inputs")),
            plugins=plugin_lines,
            plugin_options=plugin_options,
            plugin_only=bool(f.get("plugin_only")),
            languages=_optional_str(f.get("languages", "")),
            match_languages=bool(f.get("match_languages")),
            instruction_filter=_optional_str(f.get("instruction_filter", "")),
            jailbreak_filter=_optional_str(f.get("jailbreak_filter", "")),
            include_fixes=_optional_str(f.get("include_fixes", "")),
            threads=threads,
            tag=_optional_str(f.get("tag", "")),
        )

    def to_cli_args(self) -> list[str]:
        """Convert validated form data into a spikee CLI argument list."""
        args: list[str] = ["generate", "--seed-folder", f"datasets/{self.seed_folder}"]

        if self.positions:
            args += ["--positions"] + self.positions
        if self.injection_delimiters:
            args += ["--injection-delimiters", self.injection_delimiters]
        if self.spotlighting_data_markers:
            args += ["--spotlighting-data-markers", self.spotlighting_data_markers]
        if self.format:
            args += ["--format", self.format]
        if self.include_system_message:
            args.append("--include-system-message")
        if self.include_standalone_inputs:
            args.append("--include-standalone-inputs")
        if self.plugins:
            args += ["--plugins"] + self.plugins
        if self.plugin_options:
            args += ["--plugin-options", self.plugin_options]
        if self.plugin_only:
            args.append("--plugin-only")
        if self.languages:
            args += ["--languages", self.languages]
        if not self.match_languages:
            args += ["--match-languages", "false"]
        if self.instruction_filter:
            args += ["--instruction-filter", self.instruction_filter]
        if self.jailbreak_filter:
            args += ["--jailbreak-filter", self.jailbreak_filter]
        if self.include_fixes:
            args += ["--include-fixes", self.include_fixes]
        if self.threads is not None and self.threads > 1:
            args += ["--threads", str(self.threads)]
        if self.tag:
            args += ["--tag", self.tag]

        return args

    @property
    def job_name(self) -> str:
        """Human-readable job name for display in the jobs list."""
        return self.seed_folder + (f" [{self.tag}]" if self.tag else "")


# ── Test form ─────────────────────────────────────────────────────────────────


@dataclass
class TestForm:
    """Validated form data for ``POST /test/run``."""

    target: str
    datasets: list[str]
    target_options: str
    judge_options: str
    threads: int
    attempts: int
    max_retries: int
    throttle: float
    attack: str
    attack_iterations: int
    attack_options: str
    attack_only: bool
    sample: float | None
    sample_seed: int
    resume: str
    tag: str

    @classmethod
    def from_form(cls, f: ImmutableMultiDict) -> TestForm:
        """Parse and validate a test form submission.

        Raises:
            FormValidationError: if any required field is missing or invalid.
        """
        target = _require(f.get("target", ""), "Target")
        datasets_raw = [d.strip() for d in f.getlist("datasets") if d.strip()]
        if not datasets_raw:
            raise FormValidationError("At least one dataset is required.")
        # Validate each dataset name against path traversal
        for ds in datasets_raw:
            if ".." in ds or "/" in ds or "\\" in ds:
                raise FormValidationError(
                    f"Dataset name {ds!r} contains invalid characters ('..', '/', or '\\\\')."
                )
        datasets = datasets_raw

        threads = _parse_positive_int(f.get("threads", "4"), "Threads", default=4) or 4
        attempts = (
            _parse_positive_int(f.get("attempts", "1"), "Attempts", default=1) or 1
        )
        max_retries = (
            _parse_positive_int(f.get("max_retries", "3"), "Max retries", default=3)
            or 3
        )
        throttle = _parse_non_negative_float(
            f.get("throttle", "0"), "Throttle", default=0.0
        )

        attack = _optional_str(f.get("attack", ""))
        attack_iterations = (
            _parse_positive_int(
                f.get("attack_iterations", "10"), "Attack iterations", default=10
            )
            or 10
        )
        attack_options = _optional_str(f.get("attack_options", ""))
        attack_only = bool(f.get("attack_only"))

        sample_str = _optional_str(f.get("sample", ""))
        sample: float | None = None
        if sample_str:
            try:
                sv = float(sample_str)
                if 0 < sv < 1:
                    sample = sv
            except ValueError:
                raise FormValidationError("Sample must be a number between 0 and 1.")

        sample_seed = (
            _parse_positive_int(f.get("sample_seed", "42"), "Sample seed", default=42)
            or 42
        )

        resume = _optional_str(f.get("resume", "no"))
        if resume not in ("auto", "no"):
            resume = "no"

        return cls(
            target=target,
            datasets=datasets,
            target_options=_optional_str(f.get("target_options", "")),
            judge_options=_optional_str(f.get("judge_options", "")),
            threads=threads,
            attempts=attempts,
            max_retries=max_retries,
            throttle=throttle,
            attack=attack,
            attack_iterations=attack_iterations,
            attack_options=attack_options,
            attack_only=attack_only,
            sample=sample,
            sample_seed=sample_seed,
            resume=resume,
            tag=_optional_str(f.get("tag", "")),
        )

    def to_cli_args(self) -> list[str]:
        """Convert validated form data into a spikee CLI argument list."""
        args: list[str] = ["test", "--target", self.target]

        for ds in self.datasets:
            args += ["--dataset", f"datasets/{ds}"]

        if self.target_options:
            args += ["--target-options", self.target_options]
        if self.judge_options:
            args += ["--judge-options", self.judge_options]
        if self.threads != 4:
            args += ["--threads", str(self.threads)]
        if self.attempts != 1:
            args += ["--attempts", str(self.attempts)]
        if self.max_retries != 3:
            args += ["--max-retries", str(self.max_retries)]
        if self.throttle > 0:
            args += ["--throttle", f"{self.throttle:g}"]

        if self.attack:
            args += ["--attack", self.attack]
            if self.attack_iterations != 10:
                args += ["--attack-iterations", str(self.attack_iterations)]
            if self.attack_options:
                args += ["--attack-options", self.attack_options]
            if self.attack_only:
                args.append("--attack-only")

        if self.sample is not None:
            args += ["--sample", str(self.sample)]
            if self.sample_seed != 42:
                args += ["--sample-seed", str(self.sample_seed)]

        if self.resume == "auto":
            args.append("--auto-resume")
        else:
            args.append("--no-auto-resume")

        if self.tag:
            args += ["--tag", self.tag]

        return args

    @property
    def job_name(self) -> str:
        """Human-readable job name for display in the jobs list."""
        ds_label = (
            self.datasets[0]
            if len(self.datasets) == 1
            else f"{len(self.datasets)} datasets"
        )
        return f"{self.target} \u2190 {ds_label}" + (
            f" [{self.tag}]" if self.tag else ""
        )
