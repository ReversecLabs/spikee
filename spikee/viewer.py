from flask import Flask, render_template

from spikee.utilities.files import read_jsonl_file, extract_resource_name
import os

VIEWER_NAME = "SPIKEE Viewer"


def create_viewer(viewer_folder, results_file, results_data) -> Flask:

    viewer = Flask(
        VIEWER_NAME,
        static_folder=os.path.join(viewer_folder, "static"),
        template_folder=os.path.join(viewer_folder, "templates"),
    )

    results_dict = {}
    for entry in results_data:
        results_dict[entry['id']] = entry

    # Context Processor (Allows templates to run functions)
    @viewer.context_processor
    def utility_processor():
        def get_app_name():
            """Return the name of the viewer application."""
            return VIEWER_NAME

        def get_result_file():
            """Return the name of the results file."""
            return results_file

        def get_results_data():
            """Return the results data."""
            return results_data

        return dict(
            get_app_name=get_app_name,
            get_result_file=get_result_file,
            get_results_data=get_results_data,
        )

    @viewer.route("/")
    def index():
        return render_template("result_file.html")

    @viewer.route("/entry/<entry>")
    def result_entry(entry):
        return render_template("result_entry.html", entry=results_dict.get(entry))

    return viewer


def run_viewer(args):
    result_file = extract_resource_name(args.result_file)
    result_data = read_jsonl_file(args.result_file)

    viewer_folder = os.path.join(os.getcwd(), "viewer")
    if not os.path.isdir(viewer_folder):
        raise FileNotFoundError(f"[Error] Viewer folder not found at {viewer_folder}, please run 'spikee init --include-viewer' to set up the viewer files.")

    viewer = create_viewer(
        viewer_folder=viewer_folder,
        results_file=result_file,
        results_data=result_data
    )

    viewer.run(
        debug=True,
        host=args.host,
        port=args.port
    )
