from flask import Blueprint, render_template

settings_bp = Blueprint("settings", __name__)


@settings_bp.route("/")
def index():
    return render_template("settings/index.html")
