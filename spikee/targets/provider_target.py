from spikee.templates.provider_target import ProviderTarget

from dotenv import load_dotenv


class ProviderTargetModule(ProviderTarget):
    pass


if __name__ == "__main__":
    load_dotenv()
    target = ProviderTargetModule()
    print("Supported provider keys:", target.get_available_option_values())
    try:

        print(target.process_input("Hello!", target_options="bedrock-claude35-sonnet"))
    except Exception as err:
        print("Error:", err)
