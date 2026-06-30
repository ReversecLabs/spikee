# spikee/viewer/blueprints/results.py
"""Results Blueprint — Phase 0 stub. Full implementation in Phase 1."""

from flask import Blueprint, render_template

results_bp = Blueprint("results", __name__)


@results_bp.route("/")
@results_bp.route("")
def index():
    return render_template("_placeholder.html", section="Results")


@results_bp.route("/overview")
def overview():
    return render_template("_placeholder.html", section="Results – Overview")


@results_bp.route("/entries")
def entries():
    return render_template("_placeholder.html", section="Results – Entries")


@results_bp.route("/entry/<entry_id>")
def entry(entry_id):
    return render_template("_placeholder.html", section="Results – Entry")
