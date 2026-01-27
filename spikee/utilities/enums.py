import enum


class Turn(enum.Enum):
    SINGLE = "single-turn"
    MULTI = "multi-turn"


class PluginType(enum.Enum):
    BASIC = "Basic"
    ATTACK_BASED = "Attack-Based"
    LLM_BASED = "LLM-Based"


class AttackType(enum.Enum):
    BASIC = "Basic"
    LLM_DRIVEN = "LLM-Driven"
    MULTI = "Multi-Turn"


class JudgeType(enum.Enum):
    BASIC = "Basic"
    LLM_BASED = "LLM-Based"
