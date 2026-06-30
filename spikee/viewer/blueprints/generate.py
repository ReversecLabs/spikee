# spikee/viewer/blueprints/generate.py
"""Generate Blueprint — Phase 0 stub. Full implementation in Phase 3."""

from flask import Blueprint, render_template

generate_bp = Blueprint("generate", __name__)


@generate_bp.route("/")
@generate_bp.route("")
def index():
    return render_template("_placeholder.html", section="Generate")


@generate_bp.route("/seeds")
def seeds():
    return render_template("_placeholder.html", section="Generate – Seeds")


@generate_bp.route("/datasets")
def datasets():
    return render_template("_placeholder.html", section="Generate – Datasets")


@generate_bp.route("/run")
def run():
    return render_template("_placeholder.html", section="Generate – Run")
