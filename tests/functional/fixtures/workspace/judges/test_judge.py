from typing import List, Optional

from spikee.templates.judge import Judge


class TestJudge(Judge):

    def get_available_option_values(self) -> List[str]:
        return ["mode=fail", "mode=success"]

    def judge(
        self,
        llm_input,
        llm_output,
        judge_args: str = "",
        judge_options: Optional[str] = None,
    ) -> bool:
        mode = self._parse_mode(judge_options)
        return mode == "success"

    @staticmethod
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
