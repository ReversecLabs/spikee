from typing import Optional


def get_available_option_values():
    return ["mode=fail", "mode=success"]


def _parse_mode(option: Optional[str]) -> str:
    if not option:
        return "fail"

    opt = option
    if ":" in opt:
        _, opt = opt.split(":", 1)
    opt = opt.strip()

    if opt.startswith("mode="):
        return opt.split("=", 1)[1].strip()

    return "fail"


def judge(
    llm_input: str,
    llm_output: str,
    judge_args: str,
    judge_options: Optional[str] = None,
) -> bool:
    mode = _parse_mode(judge_options)
    return mode == "success"
