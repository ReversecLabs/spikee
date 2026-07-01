from flask import Blueprint, current_app, jsonify, render_template, request

settings_bp = Blueprint("settings", __name__)


@settings_bp.route("/")
def index():
    return render_template("settings/index.html")


@settings_bp.route("/api/refresh-modules", methods=["POST"])
def refresh_modules():
    """Refresh the module cache by resetting and restarting warm_cache."""
    from spikee.viewer.blueprints import _cache as _module_cache

    with _module_cache._store_lock:
        _module_cache._store.clear()
        for t in _module_cache._type_ready:
            _module_cache._type_ready[t] = False
    _module_cache._all_ready.clear()

    import threading
    threading.Thread(
        target=_module_cache.warm_cache, daemon=True, name="module-cache-warmer"
    ).start()

    return jsonify({"status": "refreshing"})


@settings_bp.route("/api/truncate", methods=["POST"])
def update_truncate():
    """Update the truncate length globally."""
    data = request.get_json()
    truncate_length = data.get("truncate_length", 500)

    if not isinstance(truncate_length, int) or truncate_length < 0:
        return jsonify({"error": "truncate_length must be a non-negative integer"}), 400

    current_app.jinja_env.globals["truncate_length"] = truncate_length
    return jsonify({"truncate_length": truncate_length})
