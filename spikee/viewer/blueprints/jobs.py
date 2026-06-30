# spikee/viewer/blueprints/jobs.py
"""Jobs Blueprint — Phase 0 stub. Full implementation in Phase 4b."""

from flask import Blueprint, render_template

jobs_bp = Blueprint("jobs", __name__)


@jobs_bp.route("/")
@jobs_bp.route("")
def index():
    return render_template("_placeholder.html", section="Jobs")


@jobs_bp.route("/<job_id>")
def detail(job_id):
    return render_template("_placeholder.html", section="Jobs – Detail")
