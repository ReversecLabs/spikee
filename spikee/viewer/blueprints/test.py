# spikee/viewer/blueprints/test.py
"""Test Blueprint — test form with real data and job submission."""

from __future__ import annotations

from flask import Blueprint, abort, redirect, render_template, request, url_for

from spikee.utilities.modules import collect_datasets, collect_modules
from spikee.viewer.blueprints._shared import module_tags as _module_tags
from spikee.viewer.job_queue import job_queue, spawn_job

test_bp = Blueprint("test", __name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_datasets() -> list[str]:
    return collect_datasets()


def _collect_modules(module_type: str) -> dict:
    """Return {local: [{name, tags, tags_text}], builtin: [...]} for optgroup rendering."""
    _all, local_names, builtin_names = collect_modules(module_type)

    def _entry(name: str) -> dict:
        tags = _module_tags(name, module_type)
        tags_text = (" " + " ".join(f"[{t['label']}]" for t in tags)) if tags else ""
        return {"name": name, "tags": tags, "tags_text": tags_text}

    return {
        "local":   [_entry(m) for m in sorted(local_names)],
        "builtin": [_entry(m) for m in sorted(builtin_names)],
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@test_bp.route("/")
@test_bp.route("")
def index():
    return redirect(url_for("test.run"))


@test_bp.route("/run", methods=["GET"])
def run():
    return render_template(
        "test/run.html",
        datasets=_collect_datasets(),
        targets=_collect_modules("targets"),
        attacks=_collect_modules("attacks"),
    )


@test_bp.route("/run", methods=["POST"])
def run_post():
    f = request.form

    # ── Required ──────────────────────────────────────────────────────────────
    target = f.get("target", "").strip()
    if not target:
        abort(400, description="target is required.")

    datasets_selected = [d.strip() for d in f.getlist("datasets") if d.strip()]
    if not datasets_selected:
        abort(400, description="At least one dataset is required.")

    args = ["test", "--target", target]

    for ds in datasets_selected:
        args += ["--dataset", f"datasets/{ds}"]

    # ── Target options ────────────────────────────────────────────────────────
    if target_opts := f.get("target_options", "").strip():
        args += ["--target-options", target_opts]

    # ── Judge ─────────────────────────────────────────────────────────────────
    if judge_opts := f.get("judge_options", "").strip():
        args += ["--judge-options", judge_opts]

    # ── Execution ─────────────────────────────────────────────────────────────
    threads = f.get("threads", "4").strip()
    try:
        if int(threads) != 4:
            args += ["--threads", threads]
    except ValueError:
        pass

    attempts = f.get("attempts", "1").strip()
    try:
        if int(attempts) != 1:
            args += ["--attempts", attempts]
    except ValueError:
        pass

    max_retries = f.get("max_retries", "3").strip()
    try:
        if int(max_retries) != 3:
            args += ["--max-retries", max_retries]
    except ValueError:
        pass

    throttle = f.get("throttle", "0").strip()
    try:
        if float(throttle) > 0:
            args += ["--throttle", throttle]
    except ValueError:
        pass

    # ── Attack ────────────────────────────────────────────────────────────────
    if attack := f.get("attack", "").strip():
        args += ["--attack", attack]

        attack_iters = f.get("attack_iterations", "10").strip()
        try:
            if int(attack_iters) != 10:
                args += ["--attack-iterations", attack_iters]
        except ValueError:
            pass

        if attack_opts := f.get("attack_options", "").strip():
            args += ["--attack-options", attack_opts]

        if f.get("attack_only"):
            args.append("--attack-only")

    # ── Sampling ──────────────────────────────────────────────────────────────
    if sample := f.get("sample", "").strip():
        try:
            s = float(sample)
            if 0 < s < 1:
                args += ["--sample", str(s)]
                seed = f.get("sample_seed", "42").strip()
                if seed != "42":
                    args += ["--sample-seed", seed]
        except ValueError:
            pass

    # ── Resume behaviour ──────────────────────────────────────────────────────
    # Always emit a resume flag — never rely on interactive stdin
    resume = f.get("resume", "no").strip()
    if resume == "auto":
        args.append("--auto-resume")
    else:
        args.append("--no-auto-resume")

    # ── Tag ───────────────────────────────────────────────────────────────────
    tag = f.get("tag", "").strip()
    if tag:
        args += ["--tag", tag]

    # ── Job name ──────────────────────────────────────────────────────────────
    ds_label = datasets_selected[0] if len(datasets_selected) == 1 else f"{len(datasets_selected)} datasets"
    name = f"{target} ← {ds_label}" + (f" [{tag}]" if tag else "")

    job = job_queue.create(type="test", name=name, args=args)
    spawn_job(job)
    return redirect(url_for("jobs.detail", job_id=job.id))
