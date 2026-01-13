from flask import Flask, render_template, send_file, abort
from io import BytesIO
import os
import base64
import hashlib
from selenium import webdriver

from spikee.utilities.files import read_jsonl_file, extract_resource_name


VIEWER_NAME = "SPIKEE Viewer"


def create_viewer(viewer_folder, results_file, results_data, host, port) -> Flask:

    viewer = Flask(
        VIEWER_NAME,
        static_folder=os.path.join(viewer_folder, "static"),
        template_folder=os.path.join(viewer_folder, "templates"),
    )

    results_dict = {}
    for entry in results_data:
        results_dict[str(entry['id'])] = entry

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

        def string_to_colour(string: str) -> str:
            """
            Convert a string to a visually distinct hex colour code.
            Ensures the same string always maps to the same colour,
            and avoids colours that are too light or too dark.
            Allows some slightly brighter colours.
            """

            # Hash the string deterministically
            hash_bytes = hashlib.md5(string.encode("utf-8")).digest()
            # Use first 3 bytes for RGB
            r, g, b = hash_bytes[0], hash_bytes[1], hash_bytes[2]

            # Clamp to avoid too-dark or too-light colours
            min_val, max_val = 80, 230  # allow slightly brighter colours
            def clamp(x): return min_val + (x % (max_val - min_val))

            r, g, b = clamp(r), clamp(g), clamp(b)
            return f'#{r:02x}{g:02x}{b:02x}'

        return dict(
            get_app_name=get_app_name,
            get_result_file=get_result_file,
            get_results_data=get_results_data,
            string_to_colour=string_to_colour
        )

    @viewer.route("/")
    def index():
        return render_template("result_file.html")

    @viewer.route("/entry/<entry>")
    def result_entry(entry):
        entry_data = results_dict.get(entry)
        if not entry_data:
            abort(404, description="Entry not found")

        return render_template("result_entry.html", entry=entry_data)

    @viewer.route("/entry/<entry>/card")
    def result_card(entry):
        entry_data = results_dict.get(entry)
        if not entry_data:
            abort(404, description="Entry not found")

        return render_template("download.html", entry=entry_data, download=True)

    @viewer.route("/entry/<entry>/download")
    def result_to_image(entry):
        # Use Selenium to render the HTML and capture a screenshot as PNG bytes, allowing JS to run
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")  # Use new headless mode for better JS support
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")

        driver = webdriver.Chrome(options=options)
        try:
            driver.get(f"http://{host}:{port}/entry/{entry}/card")  # Dummy URL
            img_bytes = driver.get_screenshot_as_png()
        finally:
            driver.quit()

        # Send as downloadable file
        return send_file(
            BytesIO(img_bytes),
            mimetype='image/png',
            as_attachment=True,
            download_name=f"{results_file}_{entry}.png"
        )

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
        results_data=result_data,
        host=args.host,
        port=args.port
    )

    viewer.run(
        debug=True,
        host=args.host,
        port=args.port
    )
