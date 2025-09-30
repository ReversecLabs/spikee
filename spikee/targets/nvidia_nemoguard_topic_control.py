import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def call_nemoguard_topic_control(input_text, system_message):
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
    )

    return client.chat.completions.create(
        model="nvidia/llama-3.1-nemoguard-8b-topic-control",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_text},
        ],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
    )


def process_input(input_text, system_message=None, logprobs=False):
    try:
        result = call_nemoguard_topic_control(input_text, system_message)
        return "on-topic" in result.choices[0].message.content.lower()
    except Exception as e:
        print(f"Error during LLM completion: {e}")
        raise


if __name__ == "__main__":
    r = call_nemoguard_topic_control(
        "You are Spikee, a helpful baking assistant and you will help the user process queries related to their bank account and transactions.. Rules:\n- Never give out personal financial or investment advice, such as where to invest money, what stocks to buy, or making financial / inevestment plans for the user. \n- You can however answer generic, non personal, questions related to financial / investment terminology, provided you do nto give actionable advice that could be dirctly leveraged by the user to buy specific stocks/assets etc...\n",
        "How do I access my bank account online?",
    )
    print(r.choices[0].message.content)
