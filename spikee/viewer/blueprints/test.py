# spikee/viewer/blueprints/test.py
"""Test Blueprint — test form with real data and job submission."""

from __future__ import annotations

from flask import Blueprint, Response, abort, redirect, render_template, request, url_for

from spikee.utilities.modules import collect_datasets, collect_modules
from spikee.utilities.modules import (
    get_description_from_module, get_options_from_module, load_module_from_path,
)
from spikee.viewer.blueprints._shared import module_tags as _module_tags
from spikee.viewer.blueprints import _cache as _module_cache
from spikee.viewer.blueprints._forms import FormValidationError, TestForm
from spikee.viewer.job_queue import job_queue, spawn_job

test_bp = Blueprint("test", __name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_datasets() -> list[str]:
    """Return a list of dataset filenames available in CWD/datasets/."""
    return collect_datasets()


def _collect_modules_for_target(module_type: str, rich: bool = False) -> dict:
    """Return {local: [{name, tags, ...}], builtin: [...]} for target/attack rendering.

    When *rich* is True, each entry also includes description, options, and
    llm_required — used by the attacks picker to show inline metadata.
    """
    _all, local_names, builtin_names = collect_modules(module_type)

    def _entry(name: str) -> dict:
        tags = _module_tags(name, module_type)
        entry: dict = {"name": name, "tags": tags}
        if rich:
            description = ""
            options: list[str] = []
            llm_required = False
            try:
                mod = load_module_from_path(name, module_type)
                desc = get_description_from_module(mod, module_type)
                if desc and isinstance(desc, tuple) and len(desc) >= 2:
                    description = str(desc[1]) if desc[1] else ""
                opts = get_options_from_module(mod, module_type)
                if opts and isinstance(opts, tuple) and len(opts) >= 2:
                    option_list, llm_req = opts
                    options = list(option_list) if option_list else []
                    llm_required = bool(llm_req)
            except Exception:
                pass
            entry["description"] = description
            entry["options"] = options
            entry["llm_required"] = llm_required
        return entry

    return {
        "local":   [_entry(m) for m in sorted(local_names)],
        "builtin": [_entry(m) for m in sorted(builtin_names)],
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@test_bp.route("/")
@test_bp.route("")
def index() -> Response:
    """Redirect to the test run page."""
    return redirect(url_for("test.run"))


@test_bp.route("/run", methods=["GET"])
def run() -> str:
    """Render the test configuration form."""
    return render_template(
        "test/run.html",
        datasets=_collect_datasets(),
    )


@test_bp.route("/partials/targets-options")
def targets_options_partial() -> str:
    """HTMX partial — returns <option>/<optgroup> HTML for the target <select>."""
    if not _module_cache.is_type_ready("targets"):
        return render_template("partials/_picker_loading.html",
                               target_id="target",
                               poll_url="/test/partials/targets-options",
                               label="targets")
    return render_template("partials/_targets_options.html",
                            targets=_collect_modules_for_target("targets"))


@test_bp.route("/partials/attacks-list")
def attacks_list_partial() -> str:
    """HTMX partial — returns attack button list HTML or a polling spinner."""
    if not _module_cache.is_type_ready("attacks"):
        return render_template("partials/_picker_loading.html",
                               target_id="attack-list",
                               poll_url="/test/partials/attacks-list",
                               label="attacks")
    return render_template("partials/_attacks_list.html",
                            attacks=_collect_modules_for_target("attacks", rich=True))


@test_bp.route("/run", methods=["POST"])
def run_post() -> Response:
    """Handle test form submission, create a job, and redirect to its detail page."""
    try:
        form = TestForm.from_form(request.form)
    except FormValidationError as exc:
        abort(400, description=str(exc))
        return  # unreachable; satisfies type checkers

    args = form.to_cli_args()
    job = job_queue.create(type="test", name=form.job_name, args=args)
    spawn_job(job)
    return redirect(url_for("jobs.detail", job_id=job.id))
