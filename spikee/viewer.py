from flask import Flask, abort, redirect, render_template, request, send_file
from io import BytesIO
# from selenium import webdriver
import ast
import hashlib
import html
import json
import os
import re
import logging

from spikee.templates.standardised_conversation import StandardisedConversation
from spikee.utilities.files import process_jsonl_input_files, read_jsonl_file, write_jsonl_file, extract_resource_name
from spikee.utilities.results import ResultProcessor, generate_query, extract_entries
from spikee.judge import call_judge


VIEWER_NAME = "SPIKEE Viewer"
TRUNCATE_LENGTH = 500


def create_viewer(viewer_folder, results_files, host, port, allow_ast=False) -> Flask:

    viewer = Flask(
        VIEWER_NAME,
        static_folder=os.path.join(viewer_folder, "static"),
        template_folder=os.path.join(viewer_folder, "templates"),
    )

    # Suppress Flask logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)


# region Helper Functions

    def highlight_headings(result_output):
        """Highlight headings in the resource processor outputs by wrapping them in <mark> and <strong> tags."""
        if not isinstance(result_output, str):
            return result_output

        def repl(match):
            heading = match.group(1)
            return f'<mark><strong>=== {heading} ===</strong></mark>'

        return re.sub(r"===\s*(.*?)\s*===", repl, result_output)

    def load_file(result_files: dict[str, str]) -> dict:
        """Load and process result files, returning combined results and processed output."""
        results = {}
        for name, result_file in result_files.items():
            entries = read_jsonl_file(result_file)

            for entry in entries:
                entry['source_file'] = result_file

                if entry['response'] is not None and entry['response'] != "":
                    backup = entry['response']
                    try:
                        entry['response'] = json.loads(entry['response'])
                    except Exception:
                        if allow_ast:
                            try:
                                entry['response'] = ast.literal_eval(entry['response'])
                            except Exception:
                                entry['response'] = backup
                        else:
                            entry['response'] = backup

                results[str(name + "-" + str(entry['id']))] = (entry)

        if len(result_files) > 1:
            result_processor = highlight_headings(ResultProcessor(results=results.values(), result_file="combined").generate_output(combined=True))

        else:
            result_processor = highlight_headings(ResultProcessor(results=results.values(), result_file=list(result_files.keys())[0]).generate_output())

        return results, result_processor

    def reload_files(result_file: str = "combined"):
        """Reload the result files based on the selected result file."""
        if result_file == "combined":
            selected_files[0] = "combined"
            loaded[0], result_processor[0] = load_file(result_files=loaded_files)
            return True

        elif result_file in loaded_files:
            selected_files[0] = result_file
            loaded[0], result_processor[0] = load_file(result_files={result_file: loaded_files[result_file]})
            return True

        else:
            return False

    def is_safe_relative_url(url):
        """Check if a URL is a safe relative URL to prevent open redirects."""
        # Only allow relative URLs that start with a single slash and do not contain a scheme or netloc
        return url and url.startswith('/') and not url.startswith('//') and ':' not in url.split('?', 1)[0]

# endregion

    # Load startup result files
    loaded_files = {extract_resource_name(f): f for f in results_files}
    selected_files = ["combined"]
    loaded = [None]
    result_processor = [None]

    reload_files(selected_files[0])

    # Context Processor (Allows templates to run functions)
    @viewer.context_processor
    def utility_processor():
        def get_app_name():
            """Return the name of the viewer application."""
            return VIEWER_NAME

        def get_loaded_results_data():
            """Return the results data."""
            return loaded[0]

        def get_result_files():
            """Return the available results files."""
            return list(loaded_files.keys())

        def get_selected_file():
            """Return the currently selected results file."""
            return selected_files[0]

        def get_result_processor():
            """Return the result processor."""
            return result_processor[0]

        def process_output(output: str, truncated: bool = False) -> str:
            """Process output string for display."""

            if output is None:
                return "—"

            elif truncated and isinstance(output, str) and len(output) > TRUNCATE_LENGTH:
                return output[:TRUNCATE_LENGTH] + "...[Truncated]"

            return output

        def process_conversation(conversation_data: str, truncated: bool = False) -> str:
            standardised_conversation = StandardisedConversation()
            standardised_conversation.add_conversation(conversation_data)

            def render_message(node, message) -> str:
                # node_entry = f'<div class="code-block h-100 result-input"><strong style="color: {string_to_colour(str("node"))};">node:</strong> {html.escape(str(node))}</div>'

                if isinstance(message['data'], dict):
                    data = [
                        f'''<div class="code-block h-100 result-input"><strong style="color: {string_to_colour(str(key))};">{html.escape(str(key))}:</strong> {html.escape(process_output(str(value), truncated))}</div>''' for key, value in message['data'].items()]

                    message = ''.join(data)
                elif isinstance(message['data'], list):
                    data = [f'''<div class="code-block h-100 result-input">{html.escape(process_output(str(item), truncated))}</div>''' for item in message['data']]

                    message = ''.join(data)
                else:
                    message = f'<div class="code-block h-100 result-input">{html.escape(process_output(str(message["data"]), truncated))}</div>'

                return f'<li class="mb-2" id={node} value={node}><div class="d-flex flex-column">' + message + '</div></li>'

            def render_node(message_id: int) -> str:
                message = standardised_conversation.get_message(message_id)

                if message['children'] == []:
                    return render_message(message_id, message)

                else:
                    children = [render_node(child_id) for child_id in message['children']]

                    return f'{render_message(message_id, message)}<ol class="ps-3 mt-2">{''.join(children)}</ol>'

            conversation_html = f'<ol class="ps-3 mt-2">{render_node(0)}</ol>'

            print("FINAL CONVERSATION HTML", conversation_html)
            return conversation_html

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
            get_selected_file=get_selected_file,
            get_result_processor=get_result_processor,
            process_output=process_output,
            process_conversation=process_conversation,
            string_to_colour=string_to_colour
        )

    @viewer.before_request
    def before_request_func():
        """Reload result file if changed in query parameters."""
        result_file_get = request.args.get('result_file', None)
        result_file_post = request.form.get('result_file', None)

        result_file = result_file_get if result_file_get is not None else result_file_post

        if result_file is not None and result_file != selected_files[0]:
            if not reload_files(result_file):
                abort(404, description="Result file not found")

        viewer.jinja_env.globals['loaded_file'] = selected_files[0]

    @viewer.route("/", methods=["GET"])
    def index():
        return render_template("overview.html")

    @viewer.route("/file/", methods=["GET"])
    def result_file():
        category = request.args.get('category', '')
        custom_search = request.args.get('custom_search', '')

        # Filter entries based on category and custom search
        try:
            custom_query = generate_query('custom', custom_search.split('|'))

        except ValueError as e:
            abort(400, description=str(e))

        matching_entries = {}
        for id, entry in loaded[0].items():

            flag = True
            if category != '' and category != 'custom':
                flag = extract_entries(entry, category)

            if flag and custom_search != '':
                flag = extract_entries(entry, 'custom', custom_query)

            if flag:
                matching_entries[id] = entry

        return render_template("result_file.html", category=category, custom_search=custom_search, entries=matching_entries, truncated=True)

    @viewer.route("/entry/<entry>", methods=["GET"])
    def result_entry(entry):
        entry_data = loaded[0].get(entry)
        if not entry_data:
            abort(404, description="Entry not found")

        return render_template("result_entry.html", id=entry, entry=entry_data)

    @viewer.route("/entry/<entry>/task", methods=["POST"])
    def tasking(entry):
        return_url = request.form.get('return_url', '')
        task_action = request.form.get('task_action', '')

        # Validate entry exists
        entry_data = loaded[0].get(entry)
        if not entry_data:
            abort(404, description="Entry not found")

        # Process task action
        jsonl_data = read_jsonl_file(entry_data['source_file'])
        entry_id = str(entry_data['id'])
        for item in jsonl_data:
            if str(item['id']) == entry_id:
                match task_action:
                    case "toggle_success":
                        # Toggle success status
                        item['success'] = not item.get('success', False)

                    case "rejudge":
                        # Rejudge individual entry
                        item['success'] = call_judge(item, item.get('response', ''))

                break

        write_jsonl_file(entry_data['source_file'], jsonl_data)

        # Reload files to reflect changes
        reload_files(selected_files[0])

        # Redirect back to the return URL or entry page
        if return_url and is_safe_relative_url(return_url):
            return redirect(return_url)
        else:
            return redirect(f"/entry/{entry}")

    @viewer.route("/entry/<entry>/card", methods=["GET"])
    def result_card(entry):
        entry_data = loaded[0].get(entry)
        if not entry_data:
            abort(404, description="Entry not found")

        return render_template("download.html", id=entry, entry=entry_data, download=True)

#    @viewer.route("/entry/<entry>/download", methods=[])
#    def result_to_image(entry):
#        return abort(404, description="Download functionality not enabled in this environment.")

        # Use Selenium to render the HTML and capture a screenshot as PNG bytes, allowing JS to run
#        options = webdriver.ChromeOptions()
#        options.add_argument("--headless=new")  # Use new headless mode for better JS support
#        options.add_argument("--disable-gpu")
#        options.add_argument("--no-sandbox")

#        driver = webdriver.Chrome(options=options)
#        try:
#            driver.get(f"http://{host}:{port}/entry/{entry}/card?result_file={selected_files[0]}")  # Dummy URL
#            img_bytes = driver.get_screenshot_as_png()
#        finally:
#            driver.quit()

        # Send as downloadable file
#        return send_file(
#            BytesIO(img_bytes),
#            mimetype='image/png',
#            as_attachment=True,
#            download_name=f"{selected_files[0]}_{entry}.png"
#        )

    return viewer


def run_viewer(args):
    results_files = process_jsonl_input_files(args.result_file, args.result_folder, ["results", "rejudge", "extract"])

    if len(results_files) == 0:
        raise ValueError("[Error] No results files provided, please specify at least one using --result-file or --result-folder.")

    print("[Overview] Analyzing the following file(s): ")
    print(" - " + "\n - ".join(results_files))

    viewer_folder = os.path.join(os.getcwd(), "viewer")
    if not os.path.isdir(viewer_folder):
        raise FileNotFoundError(f"[Error] Viewer folder not found at {viewer_folder}, please run 'spikee init --include-viewer' to set up the viewer files.")

    viewer = create_viewer(
        viewer_folder=viewer_folder,
        results_files=results_files,
        host=args.host,
        port=args.port,
        allow_ast=args.allow_ast
    )

    viewer.run(
        debug=False,
        host=args.host,
        port=args.port
    )
