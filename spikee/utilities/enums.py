import enum


class Turn(enum.Enum):
    SINGLE = "single-turn"
    MULTI = "multi-turn"


class ModuleTag(enum.Enum):
    """Enumeration for module tags used to categorize modules."""
    # Turn-based tags
    MULTI = "Multi-Turn"
    SINGLE = "Single-Turn"

    # Models
    LLM = "LLM"
    ML = "ML"

    # Plugin Categories
    ATTACK_BASED = "Attack-Based"
    ENCODING = "Encoding"
    FORMATTING = "Formatting"
    OBFUSCATION = "Obfuscation"
    SOCIAL_ENGINEERING = "Social Engineering"
    TRANSLATION = "Translation"

def formatting_priority(tag: ModuleTag) -> int:
    """Determine the priority of a plugin based on its tags for formatting purposes."""
    match tag:
        case ModuleTag.ENCODING | ModuleTag.FORMATTING | ModuleTag.OBFUSCATION | ModuleTag.SOCIAL_ENGINEERING | ModuleTag.TRANSLATION:
            return 1
        
        case ModuleTag.SINGLE | ModuleTag.MULTI:
            return 2

        case ModuleTag.LLM | ModuleTag.ML:
            return 3
        
        case _:
            return 4

def module_tag_to_colour(tag: ModuleTag) -> str:
    tag_colour_map = {
        ModuleTag.MULTI: "magenta",
        ModuleTag.SINGLE: "white",

        ModuleTag.LLM: "yellow",
        ModuleTag.ML: "yellow",

        ModuleTag.ATTACK_BASED: "red",
        ModuleTag.ENCODING: "white",
        ModuleTag.FORMATTING: "white",
        ModuleTag.OBFUSCATION: "white",
        ModuleTag.SOCIAL_ENGINEERING: "white",
        ModuleTag.TRANSLATION: "white",
    }
    return tag_colour_map.get(tag, "white")

