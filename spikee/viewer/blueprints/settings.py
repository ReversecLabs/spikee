from __future__ import annotations

from flask import Blueprint, Response, current_app, jsonify, render_template, request

settings_bp = Blueprint("settings", __name__)



@settings_bp.route("/")
def index() -> str:
    """Render the settings page."""
    return render_template("settings/index.html")


@settings_bp.route("/api/refresh-modules", methods=["POST"])
def refresh_modules() -> Response:
    """Refresh the module cache by resetting and restarting warm_cache."""
    from spikee.viewer.blueprints import _cache as _module_cache

    _module_cache.reset_cache()

    import threading
    threading.Thread(
        target=_module_cache.warm_cache, daemon=True, name="module-cache-warmer"
    ).start()

    return jsonify({"status": "refreshing"})


@settings_bp.route("/api/truncate", methods=["POST"])
def update_truncate() -> Response | tuple[Response, int]:
    """Update the truncate length globally."""
    data = request.get_json()
    truncate_length = data.get("truncate_length", 500)

    if not isinstance(truncate_length, int) or truncate_length < 0:
        return jsonify({"error": "truncate_length must be a non-negative integer"}), 400

    current_app.jinja_env.globals["truncate_length"] = truncate_length
    return jsonify({"truncate_length": truncate_length})
