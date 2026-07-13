# spikee/viewer/app.py
"""
Spikee Web UI — Flask application factory for the Spikee Web UI.

Creates the Flask app, validates the workspace, registers all blueprints,
and provides the root landing page.
"""

from __future__ import annotations

import logging
import os
import secrets
import sys
from pathlib import Path

from flask import Flask, abort, g, render_template, request, session

from spikee.viewer.blueprints.results import results_bp
from spikee.viewer.blueprints.generate import generate_bp
from spikee.viewer.blueprints.test import test_bp
from spikee.viewer.blueprints.jobs import jobs_bp
from spikee.viewer.blueprints.settings import settings_bp
from spikee.viewer.blueprints import _cache as _module_cache
from spikee.viewer.job_queue import init_job_queue


# Directories that must exist in CWD for it to be considered a valid workspace
_WORKSPACE_MARKERS = ("datasets", "results", "targets", "attacks", "plugins")


def _validate_workspace(cwd: Path) -> None:
    """Abort with a clear message if CWD is not a valid Spikee workspace."""
    # Detect accidental launch from inside the spikee source / install root.
    # A pyproject.toml in CWD is a strong signal that this is the package source
    # tree rather than a user workspace.
    if (cwd / "pyproject.toml").is_file():
        print(
            f"ERROR: You are running 'spikee viewer' from the Spikee source directory.\n"
            f"Current directory: {cwd}\n\n"
            f"Please change to your workspace directory and run the viewer from there.\n"
            f"Example:\n"
            f"  cd ~/workspace\n"
            f"  spikee init\n"
            f"  spikee viewer",
            file=sys.stderr,
        )
        sys.exit(1)

    if not any((cwd / marker).is_dir() for marker in _WORKSPACE_MARKERS):
        print(
            f"ERROR: The current directory does not appear to be a Spikee workspace.\n"
            f"Expected at least one of: {', '.join(_WORKSPACE_MARKERS)}\n"
            f"Current directory: {cwd}\n\n"
            f"Run 'spikee init' from your workspace directory first,\n"
            f"then launch the viewer from there.",
            file=sys.stderr,
        )
        sys.exit(1)


def create_app(truncate_length: int = 500, db_path: str | None = None) -> Flask:
    """Create and configure the Flask application."""

    viewer_dir = Path(__file__).parent

    app = Flask(
        __name__,
        static_folder=str(viewer_dir / "static"),
        template_folder=str(viewer_dir / "templates"),
    )

    app.secret_key = secrets.token_hex(32)

    # Suppress noisy werkzeug request logs (keep warnings/errors)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    from spikee import __version__ as _spikee_version

    # Jinja globals available in all templates
    app.jinja_env.globals.update(
        app_name="SPIKEE",
        truncate_length=truncate_length,
        spikee_version=_spikee_version,
    )

    # CSRF protection
    @app.before_request
    def _csrf_protect():
        if "_csrf_token" not in session:
            session["_csrf_token"] = secrets.token_hex(32)
        g.csrf_token = session["_csrf_token"]
        if request.method == "POST":
            token = request.form.get("_csrf_token") or request.headers.get(
                "X-CSRFToken", ""
            )
            if not token or token != session.get("_csrf_token", ""):
                abort(403, description="Invalid CSRF token.")

    app.jinja_env.globals["csrf_token"] = lambda: g.csrf_token

    # Initialise job queue (with optional DB persistence) before blueprints
    # so that all blueprint imports reference the updated singleton.
    init_job_queue(db_path=db_path)

    # Register blueprints
    app.register_blueprint(results_bp, url_prefix="/results")
    app.register_blueprint(generate_bp, url_prefix="/generate")
    app.register_blueprint(test_bp, url_prefix="/test")
    app.register_blueprint(jobs_bp, url_prefix="/jobs")
    app.register_blueprint(settings_bp, url_prefix="/settings")

    # Root route — Spikee landing page
    @app.route("/")
    def home() -> str:
        """Render the Spikee landing page."""
        return render_template("home.html")

    # Cache status API — used by HTMX partials to detect readiness
    @app.route("/api/cache/status")
    def cache_status():
        """Return the current module cache warming status as JSON."""
        from flask import jsonify

        return jsonify(_module_cache.status_dict())

    # Start background module-tag cache warmer
    import threading as _threading

    _threading.Thread(
        target=_module_cache.warm_cache, daemon=True, name="module-cache-warmer"
    ).start()

    return app


class Viewer:
    """
    Thin wrapper that validates the workspace, creates the Flask app,
    and provides a `run()` method called from cli.py.
    """

    def __init__(self, args) -> None:
        _validate_workspace(Path(os.getcwd()))
        self.app = create_app(truncate_length=args.truncate, db_path=args.database)
        self._args = args

    def run(self) -> None:
        """Start the Flask development server with the configured host/port/debug settings."""
        self.app.run(
            debug=self._args.debug,
            host=self._args.host,
            port=self._args.port,
            # Use_reloader=False prevents the subprocess reader threads from
            # being duplicated when Flask's reloader forks the process.
            use_reloader=self._args.debug,
            threaded=True,
        )
