from flask import Flask, render_template, send_file, abort, url_for, request
from io import BytesIO
import os
import json
import hashlib
from selenium import webdriver

from spikee.utilities.files import process_jsonl_input_files, read_jsonl_file, extract_resource_name


VIEWER_NAME = "SPIKEE Viewer"


def create_viewer(viewer_folder, results, host, port) -> Flask:

    viewer = Flask(
        VIEWER_NAME,
        static_folder=os.path.join(viewer_folder, "static"),
        template_folder=os.path.join(viewer_folder, "templates"),
    )

    loaded_file = [list(results.keys())[0]] if results else []
    loaded_results = {}
    for name, entries in results.items():
        loaded_results[name] = {}
        for entry in entries:
            backup = entry['response']
            try:
                entry['response'] = json.loads(entry['response'])
            except json.JSONDecodeError:
                entry['response'] = backup

            loaded_results[name][str(entry['id'])] = entry

    # Context Processor (Allows templates to run functions)
    @viewer.context_processor
    def utility_processor():
        def get_app_name():
            """Return the name of the viewer application."""
            return VIEWER_NAME

        def get_loaded_result_file():
            """Return the name of the results file."""
            return loaded_file[0]

        def get_loaded_results_data(name: str):
            """Return the results data."""
            return loaded_results.get(name, {}).values()

        def get_result_files():
            """Return the available results files."""
            return list(loaded_results.keys())

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
            get_loaded_result_file=get_loaded_result_file,
            get_loaded_results_data=get_loaded_results_data,
            get_result_files=get_result_files,
            string_to_colour=string_to_colour
        )

    @viewer.route("/")
    def index():
        return render_template("overview.html")

    @viewer.route("/file/")
    def result_file():
        return render_template("result_file.html")

    @viewer.route("/entry/<entry>")
    def result_entry(entry):
        entry_data = loaded_results.get(loaded_file[0], {}).get(entry)
        if not entry_data:
            abort(404, description="Entry not found")

        return render_template("result_entry.html", entry=entry_data)

    @viewer.route("/load_results/")
    def load_results():
        name = request.args.get("result_file")
        if name not in results:
            abort(404, description="Results file not found")

        loaded_file[0] = name
        return url_for("index")

    @viewer.route("/entry/<entry>/card")
    def result_card(entry):
        entry_data = loaded_results.get(loaded_file[0], {}).get(entry)
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
            download_name=f"{loaded_file}_{entry}.png"
        )

    return viewer


def run_viewer(args):
    results_files = process_jsonl_input_files(args.result_file, args.result_folder, ["results", "rejudge", "extract"])

    if len(results_files) == 0:
        raise ValueError("[Error] No results files provided, please specify at least one using --result-file or --result-folder.")

    results = {}
    for result_file in results_files:
        name = extract_resource_name(result_file)
        results[name] = read_jsonl_file(result_file)

        print(f"[Info] Loaded {len(results[name])} entries from results file: {result_file}")

    viewer_folder = os.path.join(os.getcwd(), "viewer")
    if not os.path.isdir(viewer_folder):
        raise FileNotFoundError(f"[Error] Viewer folder not found at {viewer_folder}, please run 'spikee init --include-viewer' to set up the viewer files.")

    viewer = create_viewer(
        viewer_folder=viewer_folder,
        results=results,
        host=args.host,
        port=args.port
    )

    viewer.run(
        debug=True,
        host=args.host,
        port=args.port
    )
