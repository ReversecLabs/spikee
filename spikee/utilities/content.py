from spikee.utilities.enums import ContentType


class Content:
    def __init__(self, content, content_type: ContentType = ContentType.TEXT):
        self.content = content
        self.content_type = content_type

    def __repr__(self):
        return f"Content(type={self.content_type}, content={self.content})"

    def __str__(self) -> str:
        return str(self.content)


class Text(Content):
    def __init__(self, content: str):
        super().__init__(content, ContentType.TEXT)


class Audio(Content):
    def __init__(self, content):
        super().__init__(content, ContentType.AUDIO)


class Image(Content):
    def __init__(self, content):
        super().__init__(content, ContentType.IMAGE)

    def base64_inline(self) -> str:
        """Return the image content as a Base64-encoded string suitable for inline embedding."""
        return f"data:image/png;base64,{self.content}"


def content_factory(content, content_type: str = "text") -> Content:
    """Factory function to create Content objects based on content type."""

    match content_type.lower():
        case "text":
            return Text(content)
        case "audio":
            return Audio(content)
        case "image":
            return Image(content)
        case _:
            raise ValueError(f"Unsupported content type: {content_type}")
