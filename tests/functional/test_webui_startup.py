import logging
import threading
from types import SimpleNamespace
from urllib.request import urlopen

from flask import Flask
from werkzeug.serving import make_server

from spikee.viewer.app import Viewer


def test_webui_reports_actual_address_without_request_logs(monkeypatch, caplog):
    app = Flask(__name__)
    app.add_url_rule("/", view_func=lambda: "ready")
    viewer = Viewer.__new__(Viewer)
    viewer.app = app
    viewer._args = SimpleNamespace(host="127.0.0.1", port=0, debug=False)
    logger = logging.getLogger("werkzeug")
    monkeypatch.setattr(logger, "level", logging.WARNING)

    def run_server(**kwargs):
        assert logger.level == logging.INFO
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 0
        assert kwargs["debug"] is False
        assert kwargs["use_reloader"] is False
        assert kwargs["threaded"] is True
        server = make_server(
            kwargs["host"], kwargs["port"], app, request_handler=kwargs["request_handler"]
        )
        try:
            server.timeout = 5
            server.log_startup()
            thread = threading.Thread(target=server.handle_request)
            thread.start()
            try:
                with urlopen(f"http://127.0.0.1:{server.port}/", timeout=5) as response:
                    assert response.read() == b"ready"
            finally:
                thread.join(timeout=6)
            assert not thread.is_alive()
            assert f"http://127.0.0.1:{server.port}" in caplog.text
            assert "GET / HTTP" not in caplog.text
        finally:
            server.server_close()

    monkeypatch.setattr(app, "run", run_server)
    with caplog.at_level(logging.INFO):
        viewer.run()
    assert logger.level == logging.WARNING
