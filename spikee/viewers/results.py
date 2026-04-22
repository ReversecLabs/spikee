from spikee.templates.viewer import Viewer
from spikee.templates.standardised_conversation import StandardisedConversation
from spikee.utilities.files import (
    process_jsonl_input_files,
    read_jsonl_file,
    write_jsonl_file,
    extract_resource_name,
)
from spikee.utilities.results import ResultProcessor, generate_query, extract_entries
from spikee.utilities.enums import ModuleTag
from spikee.judge import call_judge

from flask import render_template, request, redirect, abort

import ast
import hashlib
import html
import json
import re
from typing import Dict, Any, Tuple, Union, List


class ResultsViewer(Viewer):
    @property
    def viewer_name(self) -> str:
        return "SPIKEE Results"

    def __init__(self, args):
        super().__init__()
        self.setup_before_request()

        # Setup initial results files
        self._initial_results_files = args.result_file, args.result_folder
        self.loaded_files = {}  # Maps resource name -> file path for currently loaded files

        self.refresh_result_files()

        # Flags
        self.allow_ast = args.allow_ast
        self.truncate_length = None if args.truncate == 0 else args.truncate

        # Initial results data
        self.selected_files = (
            "combined"  # Current result data source ("combined" or specific resource)
        )
        self.loaded = {}  # Loaded result data
        self.results_processor = None  # Loaded result processor output

        self.update_result_data(resource=self.selected_files)

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [], "Viewer for analyzing and rejudging Spikee results."

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        return [], False

    # region Results Processing

    def refresh_result_files(self) -> None:
        """Process the initial results files/folders and update the list of results files."""
        results_files = process_jsonl_input_files(
            self._initial_results_files[0],
            self._initial_results_files[1],
            ["results", "rejudge", "extract"],
        )

        if len(results_files) == 0:
            raise ValueError(
                "[Error] No results files provided, please specify at least one using --result-file or --result-folder."
            )

        print("[Overview] Analyzing the following file(s): ")
        print(" - " + "\n - ".join(results_files))

        self.loaded_files = {}
        resource_names = list()
        for file in results_files:
            resource_name = extract_resource_name(file)

            self.loaded_files[resource_name] = file
            resource_names.append(resource_name)

        resource_names.sort()

        self.app.jinja_env.globals["result_files"] = resource_names

    def refresh_result_data(
        self, result_files: Dict[str, str]
    ) -> Tuple[Dict[str, Any], ResultProcessor]:
        """Load and process result files, returning combined results and processed output."""

        results = {}
        for name, result_file in result_files.items():
            entries = read_jsonl_file(result_file)

            for entry in entries:
                entry["source_file"] = result_file

                if entry["response"] is not None and entry["response"] != "":
                    backup = entry["response"]
                    # First try to parse as JSON, then as a Python literal (if allow_ast is True), otherwise keep as string
                    try:
                        entry["response"] = json.loads(entry["response"])
                    except Exception:
                        if self.allow_ast:
                            try:
                                entry["response"] = ast.literal_eval(entry["response"])
                            except Exception:
                                entry["response"] = backup
                        else:
                            entry["response"] = backup

                results[str(name + "-" + str(entry["id"]))] = entry

        if len(result_files) > 1:
            result_processor = self.highlight_resource_headings(
                ResultProcessor(
                    results=results.values(), result_file="combined"
                ).generate_output(combined=True)
            )

        else:
            result_processor = self.highlight_resource_headings(
                ResultProcessor(
                    results=results.values(), result_file=list(result_files.keys())[0]
                ).generate_output()
            )

        return results, result_processor

    def update_result_data(self, resource: str = "combined") -> bool:
        """Update the current result data based on the specified result file(s)."""

        if resource == "combined":
            self.selected_files = "combined"
            self.loaded, self.results_processor = self.refresh_result_data(
                result_files=self.loaded_files
            )

        elif resource in self.loaded_files:
            self.selected_files = resource
            self.loaded, self.results_processor = self.refresh_result_data(
                result_files={resource: self.loaded_files[resource]}
            )

        else:
            return False

        self.app.jinja_env.globals["selected_files"] = self.selected_files
        self.app.jinja_env.globals["loaded_result_data"] = self.loaded
        self.app.jinja_env.globals["loaded_result_processor"] = self.results_processor

        return True

    # endregion

    # region Formatting Helpers

    def highlight_resource_headings(self, result_output: str):
        """Highlight headings in the resource processor outputs by wrapping them in <mark> and <strong> tags."""

        def repl(match):
            heading = match.group(1)
            return f"<mark><strong>=== {heading} ===</strong></mark>"

        return re.sub(r"===\s*(.*?)\s*===", repl, result_output)

    def truncate(self, text: str) -> str:
        if self.truncate_length is not None and len(text) > self.truncate_length:
            return text[: self.truncate_length] + "...[Truncated]"
        return text

    def process_text(self, text: Union[str, None], truncated: bool = False) -> str:
        """Process text for display."""

        if text is None:
            return "—"

        elif truncated:
            return self.truncate(text)

        return text

    def process_standardised_conversation(
        self, conversation_data: str, truncated: bool = False
    ) -> str:
        """Process a standardised conversation for display."""
        try:
            conversation = StandardisedConversation()
            conversation.add_conversation(conversation_data)

        except json.JSONDecodeError:  # return processed raw text
            return self.process_text(conversation_data, truncated=truncated)

        def render_message(node, message) -> str:
            """Render a single message as an HTML list item, including its content and metadata."""
            # node_entry = f'<div class="code-block h-100 result-input"><strong style="color: {string_to_colour(str("node"))};">node:</strong> {html.escape(str(node))}</div>'

            if isinstance(message["data"], dict):
                data = [
                    f"""<div class="code-block h-100 result-input"><strong style="color: {self.text_to_colour(str(key))};">{html.escape(str(key))}:</strong> {html.escape(self.process_text(str(value), truncated))}</div>"""
                    for key, value in message["data"].items()
                ]

                message = "".join(data)
            elif isinstance(message["data"], list):
                data = [
                    f"""<div class="code-block h-100 result-input">{html.escape(self.process_text(str(item), truncated))}</div>"""
                    for item in message["data"]
                ]
                message = "".join(data)
            else:
                message = f"""<div class="code-block h-100 result-input">{html.escape(self.process_text(str(message["data"]), truncated))}</div>"""

            return f"""<li class="mb-2" id={node} value={node}><div class="d-flex flex-column">{message}</div></li>"""

        def render_node(message_id: int) -> str:
            """Recursively render a message and its children as nested HTML lists."""
            message = conversation.get_message(message_id)

            if message["children"] == []:
                return render_message(message_id, message)

            else:
                children = [render_node(child_id) for child_id in message["children"]]

                return f"""{render_message(message_id, message)}<ol class="ps-3 mt-2">{"".join(children)}</ol>"""

        conversation_html = f"""<ol class="ps-3 mt-2">{render_node(0)}</ol>"""
        return conversation_html

    def text_to_colour(self, text: str) -> str:
        """
        Convert a string to a visually distinct hex colour code.
        Ensures the same string always maps to the same colour,
        and avoids colours that are too light or too dark.
        Allows some slightly brighter colours.
        """

        # Hash the string deterministically
        hash_bytes = hashlib.md5(text.encode("utf-8")).digest()
        # Use first 3 bytes for RGB
        r, g, b = hash_bytes[0], hash_bytes[1], hash_bytes[2]

        # Clamp to avoid too-dark or too-light colours
        min_val, max_val = 80, 230  # allow slightly brighter colours

        def clamp(x):
            return min_val + (x % (max_val - min_val))

        r, g, b = clamp(r), clamp(g), clamp(b)
        return f"#{r:02x}{g:02x}{b:02x}"

    # endregion

    @property
    def context_processor(self):
        return dict(
            process_text=self.process_text,
            process_standardised_conversation=self.process_standardised_conversation,
            text_to_colour=self.text_to_colour,
        )

    def setup_before_request(self):
        @self.app.before_request
        def before_request():
            """Reload result files, if changed in dropdown."""

            result_file_get = request.args.get("result_file", None)
            result_file_post = request.form.get("result_file", None)

            result_file = result_file_get or result_file_post

            if result_file is not None and result_file != self.selected_files:
                if not self.update_result_data(resource=result_file):
                    self.refresh_result_files()

                    description = f"<h1>File Not Found</h1> Result file '{result_file}' not found among loaded files:"
                    for loaded_file in self.loaded_files.keys():
                        description += f"<br> - {loaded_file}"

                    description += "<br><br> <a href='/'>Return to overview</a>"

                    return description, 404

            self.app.jinja_env.globals["full_path"] = request.full_path

    def setup_routes(self):
        @self.app.route("/", methods=["GET"])
        def overview():
            return render_template("overview.html")

        @self.app.route("/task", methods=["POST"])
        def task():
            return_url = request.form.get("return_url", "")
            task_action = request.form.get("task_action", "")

            print(return_url, task_action)

            match task_action:
                case "file_refresh":
                    self.refresh_result_files()

                    if (
                        self.selected_files in list(self.loaded_files.keys())
                        or self.selected_files == "combined"
                    ):
                        self.update_result_data(resource=self.selected_files)

                    else:
                        self.update_result_data(resource="combined")

                case "entry_refresh":
                    self.update_result_data(resource=self.selected_files)

            if return_url:
                return redirect(return_url)
            else:
                return redirect("/")

        @self.app.route("/file/", methods=["GET"])
        def entries():
            category = request.args.get("category", "")
            custom_search = request.args.get("custom_search", "")

            # Filter entries based on category and custom search
            try:
                custom_query = generate_query("custom", custom_search.split("|"))

                matching_entries = {}
                for eid, entry in self.loaded.items():
                    flag = True
                    if category != "" and category != "custom":
                        flag = extract_entries(entry, category)

                    if flag and custom_search != "":
                        flag = extract_entries(entry, "custom", custom_query)

                    if flag:
                        matching_entries[eid] = entry

                return render_template(
                    "result_file.html",
                    category=category,
                    custom_search=custom_search,
                    entries=matching_entries,
                    truncated=True,
                )

            except ValueError as e:
                abort(400, description=str(e))

        @self.app.route("/entry/<entry>", methods=["GET"])
        def entry(entry):
            entry_data = self.loaded.get(entry)
            if not entry_data:
                abort(404, description="Entry not found")

            else:
                return render_template("result_entry.html", id=entry, entry=entry_data)

        @self.app.route("/entry/<entry>/task", methods=["POST"])
        def entry_task(entry):
            return_url = request.form.get("return_url", "")
            task_action = request.form.get("task_action", "")

            # Validate entry exists
            entry_data = self.loaded.get(entry, None)
            if entry_data is None:
                abort(404, description="Entry not found")

            else:
                # Process task action
                jsonl_data = read_jsonl_file(entry_data["source_file"])
                entry_id = str(entry_data["id"])
                for item in jsonl_data:
                    if str(item["id"]) == entry_id:
                        match task_action:
                            case "toggle_success":
                                # Toggle success status
                                item["success"] = not item.get("success", False)

                            case "rejudge":
                                # Rejudge individual entry
                                item["success"] = call_judge(
                                    item, item.get("response", "")
                                )

                        break

                write_jsonl_file(entry_data["source_file"], jsonl_data)

                # Reload files to reflect changes
                self.update_result_data(self.selected_files)

                if return_url:
                    return redirect(return_url)
                else:
                    return redirect(f"/entry/{entry}")
