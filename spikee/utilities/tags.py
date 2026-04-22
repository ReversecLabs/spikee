import re


def validate_tag(tag):
    """
    Validates that a tag is safe to use in a filename.

    Args:
        tag (str): The tag to validate

    Returns:
        tuple: (is_valid, error_message)
            - is_valid (bool): True if tag is valid, False otherwise
            - error_message (str): Reason for validation failure or None if valid
    """
    if tag is None:
        return True, None

    # Check for empty string after stripping whitespace
    if len(tag.strip()) == 0:
        return False, "Tag cannot be empty or whitespace only"

    # Check length (reasonable max length for filename component)
    MAX_LENGTH = 50
    if len(tag) > MAX_LENGTH:
        return False, f"Tag exceeds maximum length of {MAX_LENGTH} characters"

    # Check for valid characters - alphanumeric, dash and underscore only
    pattern = re.compile(r"^[a-zA-Z0-9_-]+$")
    if not pattern.match(tag):
        return (
            False,
            "Tag can only contain letters, numbers, dash (-) and underscore (_)",
        )

    return True, None


def validate_and_get_tag(tag):
    if not tag:
        return None
    valid, err = validate_tag(tag)
    if not valid:
        print(f"Error: Invalid tag: {err}")
        exit(1)
    return tag
