# spikee/viewer/app.py
"""
SpikeeApp — Flask application factory for the Spikee web viewer.

Creates the Flask app, validates the workspace, registers all blueprints,
and provides the root landing page.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from flask import Flask, render_template

from spikee.viewer.blueprints.results import results_bp
from spikee.viewer.blueprints.generate import generate_bp
from spikee.viewer.blueprints.test import test_bp
from spikee.viewer.blueprints.jobs import jobs_bp


# Directories that must exist in CWD for it to be considered a valid workspace
_WORKSPACE_MARKERS = ("datasets", "results", "targets")


def _validate_workspace(cwd: Path) -> None:
    """Abort with a clear message if CWD is not a valid Spikee workspace."""
    if not any((cwd / marker).is_dir() for marker in _WORKSPACE_MARKERS):
        print(
            "\n[Error] The current directory does not appear to be a Spikee workspace.\n"
            f"        Expected at least one of: {', '.join(_WORKSPACE_MARKERS)}\n"
            f"        Current directory: {cwd}\n\n"
            "        Run 'spikee init' from your workspace directory first,\n"
            "        then launch the viewer from there.\n"
        )
        sys.exit(1)


def create_app(truncate_length: int = 500) -> Flask:
    """Create and configure the Flask application."""

    viewer_dir = Path(__file__).parent

    app = Flask(
        __name__,
        static_folder=str(viewer_dir / "static"),
        template_folder=str(viewer_dir / "templates"),
    )

    # Suppress noisy werkzeug request logs (keep warnings/errors)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    from spikee import __version__ as _spikee_version

    # Jinja globals available in all templates
    app.jinja_env.globals.update(
        app_name="SPIKEE",
        truncate_length=truncate_length,
        spikee_version=_spikee_version,
    )

    # Register blueprints
    app.register_blueprint(results_bp,  url_prefix="/results")
    app.register_blueprint(generate_bp, url_prefix="/generate")
    app.register_blueprint(test_bp,     url_prefix="/test")
    app.register_blueprint(jobs_bp,     url_prefix="/jobs")

    # Root route — Spikee landing page
    @app.route("/")
    def home():
        return render_template("home.html")

    return app


class Viewer:
    """
    Thin wrapper that validates the workspace, creates the Flask app,
    and provides a `run()` method called from cli.py.
    """

    def __init__(self, args):
        _validate_workspace(Path(os.getcwd()))
        self.app = create_app(truncate_length=args.truncate)
        self._args = args

    def run(self):
        self.app.run(
            debug=self._args.debug,
            host=self._args.host,
            port=self._args.port,
            # Use_reloader=False prevents the subprocess reader threads from
            # being duplicated when Flask's reloader forks the process.
            use_reloader=self._args.debug,
            threaded=True,
        )
