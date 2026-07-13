# spikee/viewer/blueprints/results.py
"""Results Blueprint — scans CWD/results/ for JSONL files and serves analysis views."""

from __future__ import annotations

import hashlib
import html as _html
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import (
    Blueprint,
    Response,
    abort,
    g,
    redirect,
    render_template,
    request,
    url_for,
)

from spikee.templates.standardised_conversation import StandardisedConversation
from spikee.utilities.files import (
    extract_resource_name,
    read_jsonl_file,
    write_jsonl_file,
)
from spikee.utilities.results import ResultProcessor, extract_entries, generate_query
from spikee.judge import call_judge

results_bp = Blueprint("results", __name__)

# ── Module-level file registry ────────────────────────────────────────────────
# Maps resource_name -> absolute Path.  Populated by scan_result_files().
loaded_files: Dict[str, Path] = {}

# ── Prefixes that count as result files ──────────────────────────────────────
_RESULT_PREFIXES = ("results", "rejudge", "extract")


# ── File scanning ─────────────────────────────────────────────────────────────


def scan_result_files() -> None:
    """
    Walk CWD/results/ (one level of subdirectories) and register every
    *.jsonl whose name starts with a recognised prefix.

    Populates the module-level `loaded_files` dict atomically to avoid
    concurrent readers seeing a partially-rebuilt dict.
    """
    global loaded_files
    new_files: Dict[str, Path] = {}

    results_dir = Path(os.getcwd()) / "results"
    if not results_dir.is_dir():
        loaded_files = new_files
        return

    def _accept(fname: str) -> bool:
        return fname.endswith(".jsonl") and any(
            fname.startswith(p) for p in _RESULT_PREFIXES
        )

    for item in sorted(results_dir.iterdir()):
        if item.is_file() and _accept(item.name):
            new_files[extract_resource_name(str(item))] = item
        elif item.is_dir():
            for child in sorted(item.iterdir()):
                if child.is_file() and _accept(child.name):
                    key = f"{item.name}/{extract_resource_name(str(child))}"
                    new_files[key] = child

    loaded_files = (
        new_files  # single atomic assignment — readers see old or new, never partial
    )


def _build_file_tree() -> List[Tuple[str, List[str]]]:
    """
    Convert loaded_files into a list of (folder, [resource_names]) tuples
    suitable for the file-selector dropdown.
    folder="" means root-level.
    """
    root: List[str] = []
    folders: Dict[str, List[str]] = {}

    for key in loaded_files:
        if "/" in key:
            folder, name = key.split("/", 1)
            folders.setdefault(folder, []).append(key)
        else:
            root.append(key)

    tree: List[Tuple[str, List[str]]] = []
    if root:
        tree.append(("", root))
    for folder in sorted(folders):
        tree.append((folder, folders[folder]))
    return tree


def _files_for_selection(selected: str) -> Dict[str, Path]:
    """
    Resolve a selection string to a dict of {resource_name: Path}.
    Selections:
      "combined"        → all files
      "folder:<name>"   → all files whose key starts with "<name>/"
      "<resource_name>" → single file
    """
    if selected == "combined":
        return dict(loaded_files)
    if selected.startswith("folder:"):
        folder = selected[len("folder:") :]
        return {k: v for k, v in loaded_files.items() if k.startswith(folder + "/")}
    if selected in loaded_files:
        return {selected: loaded_files[selected]}
    return {}


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_result_data(
    files: Dict[str, Path],
) -> Tuple[Dict[str, Any], str, Any]:
    """
    Load and combine JSONL files into an entries dict + processor output string.
    Returns: (entries_dict, processor_output_html, ResultProcessor)

    Results are memoised on Flask's ``g`` for the duration of the current request
    so that multiple routes reading the same files don't re-parse them.
    """
    # Build a stable, short cache key from the sorted file paths
    cache_key = (
        "_lrd_"
        + hashlib.md5(
            "_".join(sorted(str(p) for p in files.values())).encode()
        ).hexdigest()
    )
    cached = getattr(g, cache_key, None)
    if cached is not None:
        return cached
    entries: Dict[str, Any] = {}

    for resource_name, path in files.items():
        rows = read_jsonl_file(str(path))
        for row in rows:
            row["source_file"] = str(path)
            # Try to parse JSON responses; keep as string on failure
            if row.get("response") not in (None, ""):
                backup = row["response"]
                try:
                    row["response"] = json.loads(row["response"])
                except Exception:
                    row["response"] = backup

            key = f"{resource_name}-{row['id']}"
            entries[key] = row

    if not entries:
        return {}, "", None

    combined = len(files) > 1
    rp = ResultProcessor(
        results=list(entries.values()),
        result_file="combined" if combined else next(iter(files)),
    )
    raw_output = rp.generate_output(combined=combined)
    processor_output = _highlight_headings(raw_output)

    result = (entries, processor_output, rp)
    setattr(g, cache_key, result)
    return result


def _highlight_headings(text: str) -> str:
    """Escape plain text and wrap known heading markers in markup for display.

    This prevents XSS attacks from LLM responses while still allowing
    formatted result output.
    """
    # Escape the plain-text output first so LLM responses can't inject HTML,
    # then safely wrap known heading markers in markup.
    escaped = _html.escape(text)
    return re.sub(
        r"===\s*(.*?)\s*===",
        lambda m: f"<mark><strong>=== {m.group(1)} ===</strong></mark>",
        escaped,
    )


# ── Formatting helpers ────────────────────────────────────────────────────────


def _get_truncate_length() -> Optional[int]:
    """Get the global truncate length from Flask app config."""
    from flask import current_app

    return current_app.jinja_env.globals.get("truncate_length")


def _process_text(text, truncated: bool = False) -> str:
    """Process text for display, applying truncation if requested."""
    if text is None:
        return "—"
    text = str(text)
    truncate_length = _get_truncate_length()
    if truncated and truncate_length:
        return _truncate(text, truncate_length)
    return text


def _truncate(text: str, length: Optional[int]) -> str:
    """Truncate text to specified length, adding ellipsis if needed."""
    if length and len(text) > length:
        return text[:length] + "...[Truncated]"
    return text


def _text_to_colour(text: str) -> str:
    """Generate a deterministic color from text using MD5 hash.

    Returns an RGB hex color string suitable for use in CSS.
    Colors are constrained to a readable range (NOT too dark, NOT too bright).
    """
    h = hashlib.md5(str(text).encode("utf-8")).digest()
    r, g, b = h[0], h[1], h[2]
    lo, hi = 80, 230

    def clamp(x):
        return lo + (x % (hi - lo))

    return f"#{clamp(r):02x}{clamp(g):02x}{clamp(b):02x}"


def _process_standardised_conversation(
    conversation_data: str, truncated: bool = False
) -> str:
    """Render conversation data as HTML with proper formatting.

    Handles both structured conversation data (dict/list) and plain strings.
    Includes XSS防护 by escaping content.
    """
    try:
        conversation = StandardisedConversation()
        conversation.add_conversation(conversation_data)
    except (json.JSONDecodeError, Exception):
        return _html.escape(_process_text(str(conversation_data), truncated))

    def render_message(node, message) -> str:
        if isinstance(message["data"], dict):
            parts = [
                f'<div class="code-block result-input">'
                f'<strong style="color:{_text_to_colour(str(k))};">'
                f"{_html.escape(str(k))}:</strong> "
                f"{_html.escape(_process_text(str(v), truncated))}</div>"
                for k, v in message["data"].items()
            ]
            body = "".join(parts)
        elif isinstance(message["data"], list):
            parts = [
                f'<div class="code-block result-input">'
                f"{_html.escape(_process_text(str(item), truncated))}</div>"
                for item in message["data"]
            ]
            body = "".join(parts)
        else:
            body = (
                f'<div class="code-block result-input">'
                f"{_html.escape(_process_text(str(message['data']), truncated))}</div>"
            )
        return (
            f'<li class="mb-2" id={node} value={node}>'
            f'<div class="d-flex flex-column">{body}</div></li>'
        )

    def render_node(message_id: int) -> str:
        msg = conversation.get_message(message_id)
        rendered = render_message(message_id, msg)
        if msg["children"]:
            children_html = "".join(render_node(c) for c in msg["children"])
            return f'{rendered}<ol class="ps-3 mt-2">{children_html}</ol>'
        return rendered

    return f'<ol class="ps-3 mt-2">{render_node(0)}</ol>'


# ── Stats extraction from ResultProcessor ────────────────────────────────────


def _extract_stats(rp: ResultProcessor) -> dict:
    """Extract statistics from ResultProcessor for display.

    Returns a dict with:
        total, successes, failures, guardrails, errors
        asr (float), gtr (float)
    """
    total = rp.total_entries
    succ = rp.successful_groups
    fail = rp.failed_groups
    guard = rp.guardrail_groups
    err = rp.error_groups

    asr = f"{rp.attack_success_rate:.1f}%"
    gtr = f"{(guard / total * 100):.1f}%" if total and guard else "0.0%"

    return dict(
        total=total,
        successes=succ,
        failures=fail,
        guardrails=guard,
        errors=err,
        asr=asr,
        gtr=gtr,
    )


def _extract_breakdowns(rp: ResultProcessor):
    """Convert ResultProcessor._breakdowns into the list-of-tuples format
    the overview template expects:
      [(title, [(label, total, successes, asr_pct[, gtr_n]), ...]), ...]
    Only include breakdowns with >1 distinct value.
    """
    # Trigger breakdown generation if not already done (generate_output calls it too)
    if not rp._breakdowns:
        rp.generate_detailed_breakdowns()

    FIELD_LABELS = {
        "plugin": "Plugin",
        "attack_name": "Attack",
        "jailbreak_type": "Jailbreak Type",
        "instruction_type": "Instruction Type",
        "task_type": "Task Type",
        "lang": "Language",
        "position": "Position",
        "injection_delimiters": "Injection Delimiters",
        "spotlighting_data_markers": "Spotlighting Markers",
        "suffix_id": "Suffix",
    }

    show_gtr = rp.guardrail_groups > 0
    sections = []

    for field, label in FIELD_LABELS.items():
        data = rp._breakdowns.get(field, {})
        if len(data) <= 1:
            continue

        rows = []
        for value, stats in data.items():
            t = stats["total"]
            s = stats["successes"]
            g = stats.get("guardrails", 0)
            asr_pct = (s / t * 100) if t else 0.0
            if show_gtr:
                rows.append((str(value), t, s, round(asr_pct, 1), g))
            else:
                rows.append((str(value), t, s, round(asr_pct, 1)))

        if rows:
            sections.append((label, rows))

    return sections


# ── Context processor (helpers available in all results templates) ─────────────


@results_bp.context_processor
def _inject_helpers():
    """Inject shared template helpers into all results blueprint templates."""

    def _source_label(entry: dict) -> str:
        """Return a short display name for an entry's source file."""
        sf = entry.get("source_file", "")
        return extract_resource_name(sf) if sf else ""

    return dict(
        process_text=_process_text,
        process_standardised_conversation=_process_standardised_conversation,
        text_to_colour=_text_to_colour,
        source_label=_source_label,
    )


# ── Routes ────────────────────────────────────────────────────────────────────


@results_bp.route("/")
@results_bp.route("")
def index() -> Response:
    """Redirect to the results overview."""
    return redirect(url_for("results.overview"))


@results_bp.route("/overview")
def overview() -> str:
    """Display aggregated statistics and analysis from result files."""
    scan_result_files()  # re-scan on every overview visit to pick up new results
    selected = request.args.get("result_file", "combined")
    files = _files_for_selection(selected)

    if not loaded_files:
        # No results directory or no files found — render with empty state
        return render_template(
            "results/overview.html",
            result_file_tree=[],
            selected_file=selected,
            stats=None,
            processor_output=None,
            breakdowns=None,
        )

    if not files:
        abort(
            404,
            description=f"Result file '{selected}' not found. Use the selector to choose a valid file or refresh to rescan.",
        )

    entries, processor_output, rp = _load_result_data(files)
    stats = _extract_stats(rp)
    breakdowns = _extract_breakdowns(rp)

    return render_template(
        "results/overview.html",
        result_file_tree=_build_file_tree(),
        selected_file=selected,
        stats=stats,
        processor_output=processor_output,
        breakdowns=breakdowns,
    )


@results_bp.route("/entries")
def entries() -> str:
    """Render a paginated, filterable list of result entries."""
    selected = request.args.get("result_file", "combined")
    custom_search = request.args.get("custom_search", "")
    try:
        per_page = max(1, min(int(request.args.get("per_page", 100)), 500))
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        abort(
            400,
            description="Invalid pagination parameters — 'page' and 'per_page' must be integers.",
        )
    files = _files_for_selection(selected)

    if not loaded_files:
        return render_template(
            "results/entries.html",
            result_file_tree=[],
            selected_file=selected,
            entries={},
            custom_search=custom_search,
            page=1,
            per_page=per_page,
            total=0,
            total_pages=1,
        )

    if not files:
        abort(
            404,
            description=f"Result file '{selected}' not found. Use the selector to choose a valid file or refresh to rescan.",
        )

    all_entries, _output, _rp = _load_result_data(files)

    # Apply search filter
    try:
        if custom_search:
            # Split on | to get OR clauses; each clause is itself a list of
            # AND terms (space-separated within a clause).
            # e.g. "success:True|guardrail:True" → two OR clauses, each with 1 term
            # e.g. "plugin:base64 lang:en"        → one clause with 2 AND terms
            or_clauses = [c.strip() for c in custom_search.split("|") if c.strip()]
            clause_queries = []
            for clause in or_clauses:
                terms = [t.strip() for t in clause.split() if t.strip()]
                clause_queries.append(generate_query("custom", terms))

            def _matches(e):
                return any(extract_entries(e, "custom", q) for q in clause_queries)

            matching = {eid: e for eid, e in all_entries.items() if _matches(e)}
        else:
            matching = all_entries
    except ValueError as exc:
        abort(400, description=str(exc))

    # Paginate
    total = len(matching)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = min(page, total_pages)
    offset = (page - 1) * per_page
    items = list(matching.items())
    page_entries = dict(items[offset : offset + per_page])

    return render_template(
        "results/entries.html",
        result_file_tree=_build_file_tree(),
        selected_file=selected,
        entries=page_entries,
        custom_search=custom_search,
        page=page,
        per_page=per_page,
        total=total,
        total_pages=total_pages,
    )


@results_bp.route("/entry/<path:entry_id>")
def entry(entry_id: str) -> str:
    """Render the detail view for a single result entry."""
    selected = request.args.get("result_file", "combined")
    files = _files_for_selection(selected)

    if not files:
        abort(
            404,
            description=f"Result file '{selected}' not found. Use the selector to choose a valid file or refresh to rescan.",
        )

    all_entries, _output, _rp = _load_result_data(files)
    entry_data = all_entries.get(entry_id)
    if entry_data is None:
        abort(404, description=f"Entry '{entry_id}' not found.")

    return render_template(
        "results/entry.html",
        result_file_tree=_build_file_tree(),
        selected_file=selected,
        id=entry_id,
        entry=entry_data,
    )


def _find_entry_in_files(entry_id: str, selected: str) -> Tuple[Dict[str, Any], str]:
    """
    Locate a single entry by its composite key without running ResultProcessor.
    Reads raw JSONL rows only — used by mutation routes (toggle, rejudge, bulk).
    Returns: (entry_row_dict, source_file_path_str)
    Aborts with 404 if not found.
    """
    files = _files_for_selection(selected)
    if not files:
        files = _files_for_selection("combined")

    # entry_id format: "<resource_name>-<row_id>"
    # Find which file the entry belongs to by matching the resource_name prefix.
    for resource_name, path in files.items():
        prefix = f"{resource_name}-"
        if not entry_id.startswith(prefix):
            continue
        row_id = entry_id[len(prefix) :]
        rows = read_jsonl_file(str(path))
        for row in rows:
            if str(row["id"]) == row_id:
                return row, str(path)

    abort(404, description=f"Entry '{entry_id}' not found.")


# ── POST endpoints ────────────────────────────────────────────────────────────


def _safe_return_url(url: str, default: str) -> str:
    """Reject external / protocol-relative redirects to prevent open-redirect attacks."""
    if url and url.startswith("/") and not url.startswith("//"):
        return url
    return default


@results_bp.route("/entry/<path:entry_id>/toggle", methods=["POST"])
def toggle(entry_id: str) -> Response:
    """Toggle the success flag of a single result entry."""
    selected = request.args.get("result_file", "combined")
    entry_data, source = _find_entry_in_files(entry_id, selected)
    rows = read_jsonl_file(source)
    target_id = str(entry_data["id"])
    for row in rows:
        if str(row["id"]) == target_id:
            row["success"] = not row.get("success", False)
            break
    write_jsonl_file(source, rows)
    return redirect(url_for("results.entry", entry_id=entry_id, result_file=selected))


@results_bp.route("/entry/<path:entry_id>/rejudge", methods=["POST"])
def rejudge(entry_id: str) -> Response:
    """Re-run the judge on a single result entry and persist the updated success flag."""
    selected = request.args.get("result_file", "combined")
    entry_data, source = _find_entry_in_files(entry_id, selected)
    rows = read_jsonl_file(source)
    target_id = str(entry_data["id"])
    for row in rows:
        if str(row["id"]) == target_id:
            row["success"] = call_judge(row, row.get("response", ""))
            break
    write_jsonl_file(source, rows)
    return redirect(url_for("results.entry", entry_id=entry_id, result_file=selected))


@results_bp.route("/bulk", methods=["POST"])
def bulk() -> Response:
    """
    Bulk action on a set of entry IDs.
    Form fields:
      action      — "rejudge" | "toggle"
      entry_ids   — repeated field, one value per selected entry key
      result_file — which file selection was active
      return_url  — where to redirect afterwards
    """
    selected = request.form.get("result_file", "combined")
    action = request.form.get("action", "")
    entry_ids = request.form.getlist("entry_ids")
    return_url = request.form.get("return_url", "/results/entries")

    if not entry_ids or action not in ("rejudge", "toggle"):
        return redirect(return_url)

    files = _files_for_selection(selected)
    if not files:
        files = _files_for_selection("combined")

    # Group entry IDs by their source file using the resource_name prefix.
    # Each entry_id has the format "<resource_name>-<row_id>".
    # This avoids running ResultProcessor just to find source file paths.
    by_source: dict[str, dict[str, None]] = defaultdict(dict)
    for eid in entry_ids:
        for resource_name, path in files.items():
            prefix = f"{resource_name}-"
            if eid.startswith(prefix):
                row_id = eid[len(prefix) :]
                by_source[str(path)][row_id] = None
                break

    for source, row_ids in by_source.items():
        rows = read_jsonl_file(source)
        for row in rows:
            if str(row["id"]) not in row_ids:
                continue
            if action == "toggle":
                row["success"] = not row.get("success", False)
            elif action == "rejudge":
                row["success"] = call_judge(row, row.get("response", ""))
        write_jsonl_file(source, rows)

    return redirect(_safe_return_url(return_url, "/results/entries"))


@results_bp.route("/refresh", methods=["POST"])
def refresh() -> Response:
    """Re-scan the results directory and redirect back."""
    scan_result_files()
    return_url = request.form.get("return_url", "/results/overview")
    return redirect(_safe_return_url(return_url, "/results/overview"))
