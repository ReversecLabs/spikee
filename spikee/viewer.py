from flask import Flask, render_template, send_file, abort, url_for, request
from io import BytesIO
import os
import json
import hashlib
from selenium import webdriver

from spikee.utilities.files import process_jsonl_input_files, read_jsonl_file, extract_resource_name
from spikee.utilities.results import ResultProcessor, generate_query, extract_entries


VIEWER_NAME = "SPIKEE Viewer"
TRUNCATE_LENGTH = 500


def create_viewer(viewer_folder, results, host, port) -> Flask:

    viewer = Flask(
        VIEWER_NAME,
        static_folder=os.path.join(viewer_folder, "static"),
        template_folder=os.path.join(viewer_folder, "templates"),
    )
    loaded_file = ["combined"]
    loaded_results = {'combined': {}}
    result_processors = {}
    combined_results = {}
    for name, entries in results.items():
        loaded_results[name] = {}

        # Attempt to parse 'response' field as JSON
        for entry in entries:
            backup = entry['response']
            try:
                entry['response'] = json.loads(entry['response'])
            except json.JSONDecodeError:
                entry['response'] = backup

            loaded_results[name][str(entry['id'])] = entry
            loaded_results['combined'][str(name + "-" + str(entry['id']))] = entry

        # Create ResultProcessor for this results file
        result_processors[name] = ResultProcessor(results=entries, result_file=name).generate_output()
    result_processors["combined"] = ResultProcessor(results=loaded_results['combined'].values(), result_file="combined").generate_output()

    # Context Processor (Allows templates to run functions)
    @viewer.context_processor
    def utility_processor():
        def get_app_name():
            """Return the name of the viewer application."""
            return VIEWER_NAME

        def get_loaded_results_data(name: str):
            """Return the results data."""
            return loaded_results.get(name, {})

        def get_result_files():
            """Return the available results files."""
            return list(loaded_results.keys())

        def get_result_processor(name: str):
            """Return the result processor output."""
            return result_processors.get(name)

        def process_output(output: str, truncated: bool = False) -> str:
            """Process output string for display."""

            if output is None:
                return "—"

            elif truncated and len(output) > TRUNCATE_LENGTH:
                return output[:TRUNCATE_LENGTH] + "...[Truncated]"

            return output

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
            get_loaded_results_data=get_loaded_results_data,
            get_result_files=get_result_files,
            get_result_processor=get_result_processor,
            process_output=process_output,
            string_to_colour=string_to_colour
        )

    @viewer.before_request
    def before_request_func():
        loaded_file[0] = request.args.get('result_file', 'combined')
        viewer.jinja_env.globals['loaded_file'] = loaded_file[0]

    @viewer.route("/")
    def index():
        return render_template("overview.html")

    @viewer.route("/file/")
    def result_file():
        category = request.args.get('category', '')
        custom_search = request.args.get('custom_search', '')

        entries = loaded_results.get(loaded_file[0], {})

        # Filter entries based on category and custom search
        try:
            custom_query = generate_query('custom', custom_search.split('|'))

        except ValueError as e:
            abort(400, description=str(e))

        matching_entries = {}
        for id, entry in entries.items():

            flag = True
            if category != '' and category != 'custom':
                flag = extract_entries(entry, category)

            if flag and custom_search != '':
                flag = extract_entries(entry, 'custom', custom_query)

            if flag:
                matching_entries[id] = entry

            entries = matching_entries

        return render_template("result_file.html", category=category, custom_search=custom_search, entries=entries, truncated=True)

    @viewer.route("/entry/<entry>")
    def result_entry(entry):
        entry_data = loaded_results.get(loaded_file[0], {}).get(entry)
        if not entry_data:
            abort(404, description="Entry not found")

        return render_template("result_entry.html", id=entry, entry=entry_data)

    @viewer.route("/entry/<entry>/card")
    def result_card(entry):
        entry_data = loaded_results.get(loaded_file[0], {}).get(entry)
        if not entry_data:
            abort(404, description="Entry not found")

        return render_template("download.html", id=entry, entry=entry_data, download=True)

    @viewer.route("/entry/<entry>/download")
    def result_to_image(entry):
        # Use Selenium to render the HTML and capture a screenshot as PNG bytes, allowing JS to run
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")  # Use new headless mode for better JS support
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")

        driver = webdriver.Chrome(options=options)
        try:
            driver.get(f"http://{host}:{port}/entry/{entry}/card?result_file={loaded_file[0]}")  # Dummy URL
            img_bytes = driver.get_screenshot_as_png()
        finally:
            driver.quit()

        # Send as downloadable file
        return send_file(
            BytesIO(img_bytes),
            mimetype='image/png',
            as_attachment=True,
            download_name=f"{loaded_file[0]}_{entry}.png"
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

    print("[Overview] Analyzing the following file(s): ")
    print(" - " + "\n - ".join(results_files))

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
