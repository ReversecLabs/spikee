# spikee/viewer/blueprints/test.py
"""Test Blueprint — Phase 0 stub. Full implementation in Phase 4."""

from flask import Blueprint, render_template

test_bp = Blueprint("test", __name__)


@test_bp.route("/")
@test_bp.route("")
def index():
    return render_template("_placeholder.html", section="Test")


@test_bp.route("/run")
def run():
    return render_template("_placeholder.html", section="Test – Run")
