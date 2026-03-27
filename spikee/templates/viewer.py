from abc import ABC, abstractmethod
from flask import Flask
import os
import logging


# TODO: merge with Module class branch


class Viewer(ABC):
    def __init__(self, args):
        viewer_folder = self.get_viewer_folder()

        self.app = Flask(
            self.viewer_name,
            static_folder=os.path.join(viewer_folder, "static"),
            template_folder=os.path.join(viewer_folder, "templates"),
        )
        logging.getLogger("werkzeug").setLevel(logging.INFO)

        # App Constants
        self.app.jinja_env.globals["app_name"] = self.viewer_name

        # Set up context processor for templates
        @self.app.context_processor
        def utility_processor():
            return self.context_processor

        self.setup_routes()

    @property
    def viewer_name(self) -> str:
        return "Viewer"

    @property
    def context_processor(self):
        return dict()

# region helpers
    def get_viewer_folder(self) -> str:
        viewer_folder = os.path.join(os.getcwd(), "viewer")
        if not os.path.isdir(viewer_folder):
            raise FileNotFoundError(
                f"[Error] Viewer folder not found at {viewer_folder}, please run 'spikee init --include-viewer' to set up the viewer files."
            )

        return viewer_folder

# endregion
    def run_viewer(self, args):
        self.app.run(debug=args.debug, host=args.host, port=args.port)

    @abstractmethod
    def setup_routes(self):
        pass
