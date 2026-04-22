"""
sample_pdf_request_target.py

This is an example HTTP(s) request target for spikee that calls an external API, based on target options.
The example URLs are fictional and meant to illustrate how to structure such a target.
Demonstrates converting the incoming text into a PDF document before sending it to the application.

Usage:
    1. Place this file in your local `targets/` folder.
    2. Run the spikee test command, pointing to this target, e.g.:
         spikee test --dataset datasets/example.jsonl --target sample_pdf_request_target
"""

from spikee.templates.target import Target
from spikee.tester import GuardrailTrigger
from spikee.utilities.enums import ModuleTag

from dotenv import load_dotenv
import json
import requests
from typing import Any, Optional, List, Tuple, Union

try:
    from fpdf import FPDF
except ImportError as e:
    raise ImportError(
        'Missing required package `fpdf2`. Please install it with: `pip install "spikee[pdf]"`'
    ) from e


class SamplePDFRequestTarget(Target):
    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [], 'Sample PDF Request Target. (Requires: `pip install "spikee[pdf]"`)'

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        """Return supported attack options; Tuple[options (default is first), llm_required]"""
        return [], False

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
    ) -> Union[str, bool, Tuple[Union[str, bool], Any]]:
        url = "https://reversec.com/api/upload_pdf"

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, input_text)
        pdf_bytes = pdf.output(dest="S").encode("latin1")

        files = {
            "file": ("document.pdf", pdf_bytes, "application/pdf"),
        }

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Please analyze the content of the uploaded PDF.",
                }
            ]
        }

        try:
            response = requests.post(
                url, files=files, data={"payload": json.dumps(payload)}, timeout=30
            )

            response.raise_for_status()
            result = response.json()
            return result.get("answer", "No answer available.")

        except requests.exceptions.RequestException as e:
            if response.status_code == 400:  # Guardrail Triggered
                raise GuardrailTrigger(f"Guardrail was triggered by the target: {e}")

            else:
                print(f"Error during HTTP request: {e}")
                raise


if __name__ == "__main__":
    load_dotenv()
    try:
        target = SamplePDFRequestTarget()
        response = target.process_input("Hello!")
        print(response)
    except Exception as err:
        print("Error:", err)
