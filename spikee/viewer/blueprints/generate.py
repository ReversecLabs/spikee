# spikee/viewer/blueprints/generate.py
"""Generate Blueprint — dataset generation form and seed/dataset browsing."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Blueprint, abort, redirect, render_template, request, url_for
import markdown as md_lib

from spikee.utilities.files import read_jsonl_file
from spikee.utilities.modules import collect_datasets, collect_modules, collect_seeds
from spikee.viewer.blueprints._shared import module_tags as _module_tags
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


def _collect_plugins() -> list[dict]:
    """Return plugins as [{name, local, tags}] dicts, local first then built-in."""
    _all, local_names, builtin_names = collect_modules("plugins")
    result = []
    for name in sorted(local_names) + sorted(builtin_names):
        is_local = name in local_names
        tags = [t for t in _module_tags(name, "plugins") if t["label"] != "Single-Turn"]
        result.append({"name": name, "local": is_local, "tags": tags})
    return result


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
def index():
    return redirect(url_for("generate.run"))


@generate_bp.route("/seeds")
def seeds():
    return render_template("generate/seeds.html", seeds=_collect_seeds_with_meta())


@generate_bp.route("/seeds/<seed_name>")
def seed_detail(seed_name):
    detail = _load_seed_detail(seed_name)
    if detail is None:
        abort(404, description=f"Seed folder '{seed_name}' not found.")
    return render_template("generate/seed_detail.html", seed_name=seed_name, detail=detail)


@generate_bp.route("/datasets")
def datasets():
    return render_template("generate/datasets.html", datasets=_collect_datasets_with_meta())


@generate_bp.route("/datasets/<path:dataset_name>")
def dataset_detail(dataset_name):
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
def run():
    return render_template(
        "generate/run.html",
        seeds=_collect_seeds_with_meta(),
        plugins=_collect_plugins(),
    )


@generate_bp.route("/run", methods=["POST"])
def run_post():
    f = request.form

    seed_folder = f.get("seed_folder", "").strip()
    if not seed_folder:
        abort(400, description="seed_folder is required.")

    # Build CLI args list
    args = ["generate", "--seed-folder", f"datasets/{seed_folder}"]

    # Positions (checkboxes — multiple values)
    positions = f.getlist("positions")
    if positions:
        args += ["--positions"] + positions

    # Injection delimiters
    if delimiters := f.get("injection_delimiters", "").strip():
        args += ["--injection-delimiters", delimiters]

    # Spotlighting data markers
    if markers := f.get("spotlighting_data_markers", "").strip():
        args += ["--spotlighting-data-markers", markers]

    # Format
    if fmt := f.get("format", "").strip():
        args += ["--format", fmt]

    # Include flags
    if f.get("include_system_message"):
        args.append("--include-system-message")
    if f.get("include_standalone_inputs"):
        args.append("--include-standalone-inputs")

    # Plugins: textarea value, one entry per line (each line may contain ~ for pipes)
    # The form encodes piped groups with ~ but the CLI expects | as the pipe separator
    # All independent plugins go as space-separated values after a single --plugins flag
    plugin_lines = [ln.strip().replace("~", "|") for ln in f.get("plugins", "").splitlines() if ln.strip()]
    if plugin_lines:
        args += ["--plugins"] + plugin_lines

    # Plugin options: textarea, convert newlines to semicolons
    if plugin_opts := ";".join(
        ln.strip() for ln in f.get("plugin_options", "").splitlines() if ln.strip()
    ):
        args += ["--plugin-options", plugin_opts]

    if f.get("plugin_only"):
        args.append("--plugin-only")

    # Filtering
    if languages := f.get("languages", "").strip():
        args += ["--languages", languages]
    if not f.get("match_languages"):       # checkbox unchecked = exclude mode
        args += ["--match-languages", "false"]
    if instr_filter := f.get("instruction_filter", "").strip():
        args += ["--instruction-filter", instr_filter]
    if jb_filter := f.get("jailbreak_filter", "").strip():
        args += ["--jailbreak-filter", jb_filter]
    if fixes := f.get("include_fixes", "").strip():
        args += ["--include-fixes", fixes]

    # Threads
    if threads := f.get("threads", "").strip():
        try:
            t = int(threads)
            if t > 1:
                args += ["--threads", str(t)]
        except ValueError:
            pass

    # Tag
    if tag := f.get("tag", "").strip():
        args += ["--tag", tag]

    job = job_queue.create(
        type="generate",
        name=seed_folder + (f" [{tag}]" if tag else ""),
        args=args,
    )
    spawn_job(job)
    return redirect(url_for("jobs.detail", job_id=job.id))

