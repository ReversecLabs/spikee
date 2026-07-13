# spikee/viewer/blueprints/test.py
"""Test Blueprint — test form with real data and job submission."""

from __future__ import annotations

import os
import threading

from flask import (
    Blueprint,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from spikee.utilities.modules import collect_datasets, collect_modules
from spikee.utilities.modules import (
    get_description_from_module,
    get_options_from_module,
    load_module_from_path,
)
from spikee.viewer.blueprints._shared import module_tags as _module_tags
from spikee.viewer.blueprints import _cache as _module_cache
from spikee.viewer.blueprints._forms import FormValidationError, TestForm
from spikee.viewer.job_queue import job_queue, spawn_job

test_bp = Blueprint("test", __name__)

# Thread-safe lock for workshop log file appends
_log_lock = threading.Lock()


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
        "local": [_entry(m) for m in sorted(local_names)],
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
        return render_template(
            "partials/_picker_loading.html",
            target_id="target",
            poll_url="/test/partials/targets-options",
            label="targets",
        )
    return render_template(
        "partials/_targets_options.html", targets=_collect_modules_for_target("targets")
    )


@test_bp.route("/partials/attacks-list")
def attacks_list_partial() -> str:
    """HTMX partial — returns attack button list HTML or a polling spinner."""
    if not _module_cache.is_type_ready("attacks"):
        return render_template(
            "partials/_picker_loading.html",
            target_id="attack-list",
            poll_url="/test/partials/attacks-list",
            label="attacks",
        )
    return render_template(
        "partials/_attacks_list.html",
        attacks=_collect_modules_for_target("attacks", rich=True),
    )


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


@test_bp.route("/workshop")
def workshop() -> str:
    """Render the test workshop — send a single prompt through a plugin pipeline to a target."""
    from spikee.viewer.blueprints.generate import _collect_plugins_detail

    return render_template(
        "test/workshop.html",
        plugins=_collect_plugins_detail(),
    )


@test_bp.route("/workshop/run", methods=["POST"])
def workshop_run() -> Response:
    """Send a single input through an optional plugin pipeline to a target.

    Expects JSON body:
        target           — target module name (required)
        target_options   — options string, e.g. "model=openai/gpt-4o"
        input_text       — text to send
        pipeline         — pipe-separated plugin names, e.g. "base64|rot13" (optional)
        plugin_options   — newline/semicolon-separated "plugin:key=val" (optional)
        exclude_patterns — newline-separated regex strings to exclude from plugin transforms
        log_enabled      — bool, whether to append this send to a results file
        log_tag          — optional tag appended to the log filename

    Returns JSON:
        {response: str, guardrail: bool, error: str|null,
         log_file: str|null, log_count: int, log_error: str|null}
    """
    from spikee.utilities.hinting import get_content
    from spikee.utilities.files import append_jsonl_entry, build_file_name
    from spikee.generator import apply_plugin, load_plugins, parse_plugin_options

    data = request.get_json(silent=True) or {}
    target_name = (data.get("target") or "").strip()
    target_options = (data.get("target_options") or "").strip() or None
    input_text = data.get("input_text", "")
    pipeline_str = (data.get("pipeline") or "").strip()
    options_raw = (data.get("plugin_options") or "").strip()
    exclude_raw = (data.get("exclude_patterns") or "").strip()
    log_enabled = bool(data.get("log_enabled", False))
    log_tag = (data.get("log_tag") or "").strip() or None

    if not target_name:
        return jsonify(
            {
                "response": None,
                "guardrail": False,
                "error": "No target specified.",
                "log_file": None,
                "log_count": 0,
                "log_error": None,
            }
        )
    if not input_text:
        return jsonify(
            {
                "response": None,
                "guardrail": False,
                "error": "Input text is empty.",
                "log_file": None,
                "log_count": 0,
                "log_error": None,
            }
        )

    # ── Apply plugin pipeline (optional) ──────────────────────────────────────
    text = input_text
    if pipeline_str:
        options_normalised = ";".join(
            ln.strip()
            for ln in options_raw.replace(";", "\n").splitlines()
            if ln.strip()
        )
        plugin_option_map = parse_plugin_options(options_normalised)
        exclude_patterns = [
            ln.strip() for ln in exclude_raw.splitlines() if ln.strip()
        ] or None

        try:
            plugins_loaded = load_plugins([pipeline_str])
        except SystemExit:
            return jsonify(
                {
                    "response": None,
                    "guardrail": False,
                    "error": f"Failed to load plugin(s): {pipeline_str}",
                    "log_file": None,
                    "log_count": 0,
                    "log_error": None,
                }
            )
        except Exception as exc:
            return jsonify(
                {
                    "response": None,
                    "guardrail": False,
                    "error": str(exc),
                    "log_file": None,
                    "log_count": 0,
                    "log_error": None,
                }
            )

        if plugins_loaded:
            plugin_name, plugin_module = plugins_loaded[0]
            try:
                results = apply_plugin(
                    plugin_name,
                    plugin_module,
                    text,
                    exclude_patterns=exclude_patterns,
                    plugin_option_map=plugin_option_map,
                )
                if results:
                    text = str(get_content(results[0]))
            except Exception as exc:
                return jsonify(
                    {
                        "response": None,
                        "guardrail": False,
                        "error": f"Plugin error: {exc}",
                        "log_file": None,
                        "log_count": 0,
                        "log_error": None,
                    }
                )

    # ── Call target ───────────────────────────────────────────────────────────
    try:
        target_mod = load_module_from_path(target_name, "targets")
    except Exception as exc:
        return jsonify(
            {
                "response": None,
                "guardrail": False,
                "error": f"Failed to load target '{target_name}': {exc}",
                "log_file": None,
                "log_count": 0,
                "log_error": None,
            }
        )

    response_str: str | None = None
    guardrail = False
    call_error: str | None = None

    try:
        call_kwargs: dict = {}
        if target_options:
            call_kwargs["target_options"] = target_options

        raw = target_mod.process_input(text, **call_kwargs)

        if isinstance(raw, tuple):
            response_str = str(get_content(raw[0]))
        else:
            response_str = str(get_content(raw))

    except Exception as exc:
        if (
            type(exc).__name__ == "GuardrailTrigger"
            or "guardrail" in type(exc).__name__.lower()
        ):
            guardrail = True
            response_str = str(exc) or "Guardrail triggered."
        else:
            call_error = str(exc)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_file_name: str | None = None
    log_count: int = 0
    log_error: str | None = None

    if log_enabled and call_error is None:
        try:
            # Resolve or create the session log file path
            log_path: str | None = session.get("workshop_log_file")
            entry_count: int = session.get("workshop_log_count", 0)

            if log_path is None:
                # Sanitise target name for use in filename (replace / and spaces)
                safe_target = target_name.replace("/", "_").replace(" ", "_")
                filename = build_file_name("results", "workshop", safe_target, log_tag)
                results_dir = os.path.join(os.getcwd(), "results")
                os.makedirs(results_dir, exist_ok=True)
                log_path = os.path.join(results_dir, filename)
                session["workshop_log_file"] = log_path
                session["workshop_log_count"] = 0
                entry_count = 0

            entry_count += 1
            session["workshop_log_count"] = entry_count

            entry = {
                "id": entry_count,
                "long_id": f"workshop_{target_name}_{entry_count}",
                "input": text,
                "response": response_str,
                "success": None,
                "guardrail": guardrail,
                "error": None,
                "target": target_name,
                "target_options": target_options,
                "plugin": pipeline_str or None,
                "task_type": "workshop",
                "jailbreak_type": None,
                "instruction_type": None,
                "document_id": None,
                "position": None,
                "spotlighting_data_markers": None,
                "injection_delimiters": None,
                "lang": None,
                "system_message": None,
                "suffix_id": None,
            }

            append_jsonl_entry(log_path, entry, _log_lock)
            log_file_name = os.path.basename(log_path)
            log_count = entry_count

        except Exception as exc:
            log_error = f"Log write failed: {exc}"

    if call_error:
        return jsonify(
            {
                "response": None,
                "guardrail": False,
                "error": call_error,
                "log_file": log_file_name,
                "log_count": log_count,
                "log_error": log_error,
            }
        )

    return jsonify(
        {
            "response": response_str,
            "guardrail": guardrail,
            "error": None,
            "log_file": log_file_name,
            "log_count": log_count,
            "log_error": log_error,
        }
    )


@test_bp.route("/workshop/new-log", methods=["POST"])
def workshop_new_log() -> Response:
    """Clear the current workshop log session so the next Send starts a new file."""
    session.pop("workshop_log_file", None)
    session.pop("workshop_log_count", None)
    return jsonify({"status": "ok"})
