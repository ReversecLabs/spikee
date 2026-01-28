import enum


class Turn(enum.Enum):
    SINGLE = "single-turn"
    MULTI = "multi-turn"


class ModuleTag(enum.Enum):
    """Enumeration for module tags used to categorize modules."""
    ATTACK_BASED = "Attack-Based"
    LLM = "LLM"
    MULTI = "Multi-Turn"
    SINGLE = "Single-Turn"


def module_tag_to_colour(tag: ModuleTag) -> str:
    tag_colour_map = {
        ModuleTag.SINGLE: "blue",
        ModuleTag.MULTI: "magenta",
        ModuleTag.LLM: "yellow",
        ModuleTag.ATTACK_BASED: "red",
    }
    return tag_colour_map.get(tag, "white")
