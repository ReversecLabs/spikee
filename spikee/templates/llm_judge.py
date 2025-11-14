from abc import abstractmethod
from typing import List
import random
import string

from .judge import Judge


class LLMJudge(Judge):
    DEFAULT_LLM_MODEL = "openai-gpt-4.1-mini"

    SUPPORTED_LLM_MODELS = [
        "openai-gpt-4.1-mini",
        "openai-gpt-4o-mini",
        "ollama-phi4-mini",
        "ollama-gemma3",
        "ollama-llama3.2",
        "llamaccp-server",
        "offline",
    ]

    DEFAULT_BASE_LLAMACPP_URL = "http://localhost:8080/"

    def get_available_option_values(self):
        """
        Returns the list of supported judge_options; first option is default.
        """
        options = [self.DEFAULT_LLM_MODEL]  # Default first
        options.extend([model for model in self.SUPPORTED_LLM_MODELS if model != self.DEFAULT_LLM_MODEL])
        return options

    def _get_llm(self, judge_options=None):
        """
        Initialize and return the appropriate LLM based on judge_options.
        """
        if judge_options not in self.SUPPORTED_LLM_MODELS:
            raise ValueError(
                f"Unsupported LLM judge option: '{judge_options}'. "
                f"Supported options: {self.get_available_option_values()}"
            )

        if judge_options.startswith("openai-"):
            from langchain_openai import ChatOpenAI

            model_name = judge_options.replace("openai-", "")
            return ChatOpenAI(
                model=model_name,
                max_tokens=8,
                temperature=0,
                timeout=None,
                max_retries=2,
            )

        elif judge_options.startswith("ollama-"):
            from langchain_ollama import ChatOllama

            model_name = judge_options.replace("ollama-", "")
            return ChatOllama(
                model=model_name,
                max_tokens=8,
                temperature=0,
                timeout=None,
                max_retries=2,
            )

        elif judge_options.startswith("llamaccp-server"):
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                base_url=self.DEFAULT_BASE_LLAMACPP_URL,
                api_key="abc",
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
        elif judge_options.startswith("offline"):
            return None

        else:
            raise ValueError(
                f"Invalid judge_options format: '{judge_options}'. Expected prefix 'openai-' or 'ollama-'."
            )
