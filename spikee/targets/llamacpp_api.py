"""
spikee/targets/llamaccp_api.py

Unified OpenAI target, extending ProviderTarget, that can invoke any supported OpenAI model based on a simple key.
"""


from spikee.templates.provider_target import ProviderTarget

from dotenv import load_dotenv


class LlamacppAPITarget(ProviderTarget):
    DEFAULT_BASE_URL = "http://localhost:8080/"

    def __init__(self):
        super().__init__(provider="llamacpp_api", default_model=self.DEFAULT_BASE_URL, models=[self.DEFAULT_BASE_URL, "http://hostname:port"])


if __name__ == "__main__":
    load_dotenv()
    target = LlamacppAPITarget()
    print(target.process_input("Hello!", target_options=""))
