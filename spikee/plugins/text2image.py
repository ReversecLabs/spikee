"""
Text to Base64 Image Plugin

This plugin transforms text into image data, with Base64 encoding.

Usage:
    spikee generate --plugins base64_image
    
Requires the Pillow library for image processing. Install with:
    pip install Pillow
"""
import base64
from io import BytesIO
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from spikee.templates.plugin import Plugin
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint, Image as ImageContent
from spikee.utilities.enums import ModuleTag


class MultiModalImage(Plugin):
    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.ENCODING, ModuleTag.IMAGE], "Transforms text into image data, with Base64 encoding. (Requires: `pip install Pillow`)"

    def get_available_option_values(self) -> ModuleOptionsHint:
        return [], False

    def transform(self,
                  content: str,
                  exclude_patterns: Optional[List[str]] = None,
                  plugin_option: Optional[str] = None
                  ) -> ImageContent:

        # Load font
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()

        max_width = 800
        padding = 20

        # Split text into lines that fit max_width
        words = content.split()
        lines = []
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            w = font.getbbox(test_line)[2] - font.getbbox(test_line)[0]
            if w + 2 * padding > max_width and line:
                lines.append(line)
                line = word
            else:
                line = test_line
        if line:
            lines.append(line)

        # Calculate image height (dynamic based on number of lines)
        line_height = font.getbbox('A')[3] - font.getbbox('A')[1] + 5
        img_height = max(100, padding * 2 + line_height * len(lines))

        # Create image and draw text
        img = Image.new('RGB', (max_width, img_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        y = padding
        for line in lines:
            draw.text((padding, y), line, fill=(0, 0, 0), font=font)
            y += line_height

        # Encode image to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return ImageContent(img_base64)


if __name__ == "__main__":
    plugin = MultiModalImage()
    sample_text = "Hello, this is a test of the Base64 Image Plugin. It converts this text into an image and encodes it in Base64 format."
    result = plugin.transform(sample_text)
    print(result)
