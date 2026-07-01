# spikee/viewer/blueprints/generate.py
"""Generate Blueprint — dataset generation form and seed/dataset browsing."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Blueprint, Response, abort, redirect, render_template, request, url_for
import markdown as md_lib

from spikee.utilities.files import read_jsonl_file
from spikee.utilities.modules import (
    collect_datasets, collect_modules, collect_seeds,
    get_description_from_module, get_options_from_module, load_module_from_path,
)
from spikee.viewer.blueprints._shared import module_tags as _module_tags
from spikee.viewer.blueprints import _cache as _module_cache
from spikee.viewer.blueprints._forms import FormValidationError, GenerateForm
from spikee.generator import resolve_seed_folder
from spikee.viewer.job_queue import job_queue, spawn_job

generate_bp = Blueprint("generate", __name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DATASET_PAGE = 100  # rows shown per dataset detail page

# Files recognised inside a seed folder, in display order.
# Maps filename → (type_key, badge_classes, label)
_SEED_FILES = {
    "jailbreaks.jsonl":             ("jailbreaks",     "bg-danger",             "Jailbreaks"),
    "instructions.jsonl":           ("instructions",   "bg-warning text-dark",  "Instructions"),
    "standalone_user_inputs.jsonl": ("standalone",     "bg-info text-dark",     "Standalone"),
    "standalone_attacks.jsonl":     ("standalone",     "bg-info text-dark",     "Standalone"),
    "base_user_inputs.jsonl":       ("documents",      "bg-primary",            "Documents"),
    "base_documents.jsonl":         ("documents",      "bg-primary",            "Documents"),
    "adv_prefixes.jsonl":           ("adv_fixes",      "bg-secondary",          "Adv Prefixes"),
    "adv_suffixes.jsonl":           ("adv_fixes",      "bg-secondary",          "Adv Suffixes"),
    "system_messages.toml":         ("system_messages","bg-secondary",          "System Messages"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_seeds_with_meta() -> list[dict]:
    """Return seed names from CWD/datasets/ as list of dicts."""
    return [{"name": s, "description": ""} for s in collect_seeds()]


def _collect_datasets_with_meta() -> list[dict]:
    """Return dataset filenames from CWD/datasets/ with entry counts."""
    datasets_dir = Path(os.getcwd()) / "datasets"
    result = []
    for name in collect_datasets():
        path = datasets_dir / name
        try:
            with open(path, encoding="utf-8") as _f:
                count = sum(1 for _ in _f)
        except OSError:
            count = None
        result.append({"name": name, "entry_count": count})
    return result


def _collect_plugins() -> dict:
    """Return plugins as {local: [{name, tags, description, options, llm_required}], builtin: [...]}."""
    _all, local_names, builtin_names = collect_modules("plugins")
    exclude = {"Single-Turn"}

    def _entry(name: str) -> dict:
        tags = [t for t in _module_tags(name, "plugins") if t["label"] not in exclude]
        description = ""
        options: list[str] = []
        llm_required = False
        try:
            mod = load_module_from_path(name, "plugins")
            desc = get_description_from_module(mod, "plugins")
            if desc and isinstance(desc, tuple) and len(desc) >= 2:
                description = str(desc[1]) if desc[1] else ""
            opts = get_options_from_module(mod, "plugins")
            if opts and isinstance(opts, tuple) and len(opts) >= 2:
                option_list, llm_req = opts
                options = list(option_list) if option_list else []
                llm_required = bool(llm_req)
        except Exception:
            pass
        return {
            "name": name,
            "tags": tags,
            "description": description,
            "options": options,
            "llm_required": llm_required,
        }

    return {
        "local":   [_entry(n) for n in sorted(local_names)],
        "builtin": [_entry(n) for n in sorted(builtin_names)],
    }


def _collect_plugins_detail() -> list[dict]:
    """Return a rich list of plugin metadata for the plugins browser page.

    Each entry is a dict with:
        name        — module name
        source      — "local" or "builtin"
        tags        — [{label, colour}] (Single-Turn excluded)
        description — plain-text description string, or "" on failure
        options     — list of option strings advertised by the module, or []
        llm_required — bool; True if any option requires an LLM call
    """
    _all, local_names, builtin_names = collect_modules("plugins")
    local_set = set(local_names)
    exclude_tags = {"Single-Turn"}

    plugins = []
    for name in sorted(_all):
        tags = [t for t in _module_tags(name, "plugins") if t["label"] not in exclude_tags]
        source = "local" if name in local_set else "builtin"

        description = ""
        options: list[str] = []
        llm_required = False
        try:
            mod = load_module_from_path(name, "plugins")
            desc = get_description_from_module(mod, "plugins")
            if desc and isinstance(desc, tuple) and len(desc) >= 2:
                description = str(desc[1]) if desc[1] else ""
            opts = get_options_from_module(mod, "plugins")
            if opts and isinstance(opts, tuple) and len(opts) >= 2:
                option_list, llm_req = opts
                options = list(option_list) if option_list else []
                llm_required = bool(llm_req)
        except Exception:
            pass

        plugins.append({
            "name":         name,
            "source":       source,
            "tags":         tags,
            "description":  description,
            "options":      options,
            "llm_required": llm_required,
        })

    return plugins


def _normalize_row(row: dict, file_type: str) -> dict:
    """Normalise a raw JSONL row to a consistent preview dict."""
    text = (
        row.get("text") or row.get("content") or row.get("document")
        or row.get("instruction") or row.get("suffix") or row.get("prefix")
        or ""
    )
    if isinstance(text, (dict, list)):
        text = str(text)
    return {
        "id":   row.get("id", "—"),
        "type": row.get("jailbreak_type") or row.get("instruction_type") or "",
        "lang": row.get("lang", ""),
        "text": str(text)[:300],
    }


def _load_seed_detail(seed_name: str) -> dict | None:
    """Read a seed folder from disk and return structured file info."""
    # Prevent path traversal: resolve and verify the path stays inside datasets/
    datasets_dir = (Path(os.getcwd()) / "datasets").resolve()
    seed_path = (datasets_dir / seed_name).resolve()
    if not str(seed_path).startswith(str(datasets_dir) + os.sep):
        return None

    try:
        # collect_seeds() returns bare names; resolve_seed_folder expects "datasets/<name>"
        folder = Path(str(resolve_seed_folder(f"datasets/{seed_name}")))
    except Exception:
        return None

    files = []
    for fname, (ftype, fbadge, flabel) in _SEED_FILES.items():
        fpath = folder / fname
        if not fpath.is_file():
            continue

        if fname.endswith(".jsonl"):
            try:
                rows = read_jsonl_file(str(fpath))
            except Exception:
                rows = []
            entries = len(rows)
            preview = [_normalize_row(r, ftype) for r in rows]
        else:
            # .toml — parse with tomllib/tomli to extract system_message entries
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore
            try:
                data = tomllib.loads(fpath.read_text(encoding="utf-8"))
                configs = data.get("configurations", [])
                entries = len(configs)
                preview = [
                    {
                        "id":   i + 1,
                        "type": "",
                        "lang": "",
                        "text": str(cfg.get("system_message", ""))[:300].replace("\n", " "),
                    }
                    for i, cfg in enumerate(configs)
                ]
            except Exception:
                entries = "\u2014"
                preview = []

        # Detect which columns actually contain data (in display order)
        _col_order = ["id", "type", "lang", "text"]
        columns = [c for c in _col_order if any(str(row.get(c, "")).strip() for row in preview)]

        # Compute pixel widths for fixed columns based on max content length.
        # 'text' fills remaining space (no fixed width).
        _ch_px = 8  # approximate px per character at small font size
        _col_min = {"id": 36, "type": 60, "lang": 42}  # absolute minimums
        col_widths = {}
        for col in columns:
            if col == "text":
                continue
            header_len = {"id": 1, "type": 4, "lang": 4}.get(col, len(col))
            max_val = max((len(str(row.get(col, ""))) for row in preview), default=0)
            px = max(max(max_val, header_len) * _ch_px + 16, _col_min.get(col, 40))
            col_widths[col] = f"{px}px"

        files.append({
            "name":      fname,
            "type":      ftype,
            "badge":     fbadge,
            "label":     flabel,
            "entries":   entries,
            "preview":   preview,
            "columns":   columns,
            "col_widths": col_widths,
        })

    readme_html = None
    readme_path = folder / "README.md"
    if readme_path.is_file():
        try:
            import re as _re
            source = readme_path.read_text(encoding="utf-8")
            readme_html = md_lib.markdown(
                source,
                extensions=["fenced_code", "tables", "nl2br"],
            )
            # Sanitise: remove dangerous elements, event handlers, javascript: URIs.
            # Strip dangerous elements entirely
            readme_html = _re.sub(
                r"<(/?)(script|iframe|object|embed|form|base|meta|link|svg|math)(\s[^>]*)?/?>",
                lambda m: f"&lt;{m.group(1)}{m.group(2)}&gt;",
                readme_html,
                flags=_re.IGNORECASE,
            )
            # Strip inline event handler attributes (on*=...)
            readme_html = _re.sub(
                r'\s+on\w+\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s>]*)',
                "",
                readme_html,
                flags=_re.IGNORECASE,
            )
            # Replace javascript: and data: URIs in href/src attributes
            readme_html = _re.sub(
                r'(href|src)\s*=\s*"(javascript|data):[^"]*"',
                r'\1="#"',
                readme_html,
                flags=_re.IGNORECASE,
            )
            readme_html = _re.sub(
                r"(href|src)\s*=\s*'(javascript|data):[^']*'",
                r"\1='#'",
                readme_html,
                flags=_re.IGNORECASE,
            )
        except Exception:
            readme_html = None

    return {"path": str(folder), "files": files, "readme_html": readme_html}


def _load_dataset_entries(dataset_name: str, page: int = 1) -> dict | None:
    """Read a dataset JSONL from CWD/datasets/ and return a paginated result."""
    # Prevent path traversal: resolve and verify path stays inside datasets/
    datasets_dir = (Path(os.getcwd()) / "datasets").resolve()
    path = (datasets_dir / dataset_name).resolve()
    if not str(path).startswith(str(datasets_dir)):
        return None  # reject traversal attempts
    if not path.is_file():
        return None
    try:
        rows = read_jsonl_file(str(path))
    except Exception:
        return None

    total       = len(rows)
    total_pages = max(1, (total + _DATASET_PAGE - 1) // _DATASET_PAGE)
    page        = max(1, min(page, total_pages))
    offset      = (page - 1) * _DATASET_PAGE
    page_rows   = rows[offset : offset + _DATASET_PAGE]

    # Normalise entries: resolve content field, keep all relevant keys
    _col_order = ["id", "jailbreak_type", "instruction_type", "lang", "plugin", "position", "content"]
    normalised = []
    for r in page_rows:
        normalised.append({
            "id":               r.get("id", "—"),
            "jailbreak_type":   r.get("jailbreak_type", ""),
            "instruction_type": r.get("instruction_type", ""),
            "lang":             r.get("lang", ""),
            "plugin":           r.get("plugin", ""),
            "position":         r.get("position", ""),
            "content":          str(r.get("content") or r.get("text") or "")[:400],
        })

    # Detect which columns have at least one non-empty value across ALL rows (not just page)
    full_sample = rows[:500]  # sample up to 500 rows for column detection
    columns = [
        c for c in _col_order
        if any(str(r.get("content") or r.get("text") or "" if c == "content" else r.get(c, "")).strip()
               for r in full_sample)
    ]

    # Compute pixel widths for fixed columns
    _col_labels = {"id": "#", "jailbreak_type": "Jailbreak", "instruction_type": "Instruction",
                   "lang": "Lang", "plugin": "Plugin", "position": "Position", "content": "Content"}
    _col_min    = {"id": 36, "lang": 42, "position": 60, "plugin": 60, "jailbreak_type": 70, "instruction_type": 80}
    _ch_px      = 8
    col_widths  = {}
    for col in columns:
        if col == "content":
            continue
        header_len = len(_col_labels[col])
        max_val = max(
            (len(str(r.get("content") or r.get("text") or "" if col == "content" else r.get(col, "")))
             for r in full_sample),
            default=0,
        )
        px = max(max(max_val, header_len) * _ch_px + 16, _col_min.get(col, 40))
        col_widths[col] = f"{px}px"

    return {
        "total":       total,
        "total_pages": total_pages,
        "page":        page,
        "entries":     normalised,
        "columns":     columns,
        "col_widths":  col_widths,
        "col_labels":  _col_labels,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@generate_bp.route("/")
@generate_bp.route("")
def index() -> Response:
    """Redirect to the generate run page."""
    return redirect(url_for("generate.run"))


@generate_bp.route("/seeds")
def seeds() -> str:
    """Render the seed folder browser."""
    return render_template("generate/seeds.html", seeds=_collect_seeds_with_meta())


@generate_bp.route("/plugins")
def plugins() -> str:
    """Render the plugin workshop — build a pipeline, enter input, see output."""
    return render_template(
        "generate/plugins.html",
        plugins=_collect_plugins_detail(),
    )


@generate_bp.route("/plugins/run", methods=["POST"])
def plugins_run() -> Response:
    """Execute a plugin pipeline against a user-supplied input text.

    Expects JSON body:
        pipeline         — pipe-separated plugin names, e.g. "base64|rot13"
        plugin_options   — semicolon/newline-separated "plugin:key=val" strings
        input_text       — text to transform
        exclude_patterns — newline-separated regex strings to exclude from transformation

    Returns JSON:
        {outputs: [str], count: int, error: str|null}
    """
    from flask import jsonify
    from spikee.generator import apply_plugin, load_plugins, parse_plugin_options

    data = request.get_json(silent=True) or {}
    pipeline_str   = (data.get("pipeline") or "").strip()
    options_raw    = (data.get("plugin_options") or "").strip()
    input_text     = data.get("input_text", "")
    exclude_raw    = (data.get("exclude_patterns") or "").strip()

    # Parse exclude patterns — one regex per non-empty line
    exclude_patterns = [ln.strip() for ln in exclude_raw.splitlines() if ln.strip()] or None

    if not pipeline_str:
        return jsonify({"outputs": [], "count": 0, "error": "No plugins specified."})
    if input_text == "":
        return jsonify({"outputs": [], "count": 0, "error": "Input text is empty."})

    # Normalise options — accept newlines or semicolons as separators
    options_normalised = ";".join(
        ln.strip() for ln in options_raw.replace(";", "\n").splitlines() if ln.strip()
    )
    plugin_option_map = parse_plugin_options(options_normalised)

    try:
        # pipeline_str is already in the |-separated format load_plugins expects
        plugins_loaded = load_plugins([pipeline_str])
    except SystemExit:
        return jsonify({"outputs": [], "count": 0, "error": f"Failed to load plugin(s): {pipeline_str}"})
    except Exception as exc:
        return jsonify({"outputs": [], "count": 0, "error": str(exc)})

    if not plugins_loaded:
        return jsonify({"outputs": [], "count": 0, "error": "No plugins loaded."})

    plugin_name, plugin_module = plugins_loaded[0]

    try:
        results = apply_plugin(
            plugin_name,
            plugin_module,
            input_text,
            exclude_patterns=exclude_patterns,
            plugin_option_map=plugin_option_map,
        )
    except Exception as exc:
        return jsonify({"outputs": [], "count": 0, "error": str(exc)})

    # Coerce Content objects to plain strings
    from spikee.utilities.hinting import get_content
    outputs = [str(get_content(r)) for r in results]

    return jsonify({"outputs": outputs, "count": len(outputs), "error": None})


@generate_bp.route("/seeds/<seed_name>")
def seed_detail(seed_name: str) -> str:
    """Render the detail view for a seed folder, showing all contained files."""
    detail = _load_seed_detail(seed_name)
    if detail is None:
        abort(404, description=f"Seed folder '{seed_name}' not found.")
    return render_template("generate/seed_detail.html", seed_name=seed_name, detail=detail)


@generate_bp.route("/datasets")
def datasets() -> str:
    """Render the dataset browser listing all generated datasets."""
    return render_template("generate/datasets.html", datasets=_collect_datasets_with_meta())


@generate_bp.route("/datasets/<path:dataset_name>")
def dataset_detail(dataset_name: str) -> str:
    """Render a paginated view of a single dataset's entries."""
    page   = max(1, int(request.args.get("page", 1)))
    result = _load_dataset_entries(dataset_name, page)
    if result is None:
        abort(404, description=f"Dataset '{dataset_name}' not found.")
    return render_template(
        "generate/dataset_detail.html",
        dataset_name=dataset_name,
        meta={"name": dataset_name, "entry_count": result["total"]},
        entries=result["entries"],
        page=result["page"],
        total_pages=result["total_pages"],
        total=result["total"],
        columns=result["columns"],
        col_widths=result["col_widths"],
        col_labels=result["col_labels"],
    )


@generate_bp.route("/run", methods=["GET"])
def run() -> str:
    """Render the dataset generation form."""
    return render_template(
        "generate/run.html",
        seeds=_collect_seeds_with_meta(),
    )


@generate_bp.route("/partials/plugins-list")
def plugins_list_partial() -> str:
    """HTMX partial — returns plugin button list HTML or a polling spinner."""
    if not _module_cache.is_type_ready("plugins"):
        return render_template("partials/_picker_loading.html",
                               target_id="plugin-list",
                               poll_url="/generate/partials/plugins-list",
                               label="plugins")
    return render_template("partials/_plugins_list.html", plugins=_collect_plugins())


@generate_bp.route("/run", methods=["POST"])
def run_post() -> Response:
    """Handle dataset generation form submission, create a job, and redirect to its detail page."""
    try:
        form = GenerateForm.from_form(request.form)
    except FormValidationError as exc:
        abort(400, description=str(exc))
        return  # unreachable; satisfies type checkers

    args = form.to_cli_args()
    job = job_queue.create(type="generate", name=form.job_name, args=args)
    spawn_job(job)
    return redirect(url_for("jobs.detail", job_id=job.id))
