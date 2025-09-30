from transformers import pipeline

pipe = pipeline("text-classification", model="leolee99/InjecGuard")


def detect_prompt_injection_result(data: list):
    """
    Detects prompt injection based on the score from the pipe result.

    Args:
        data (list): Input data to be processed by the pipe.
        threshold (float): The score threshold to classify as injection.

    Returns:
        bool: True if the score is greater than or equal to the threshold, False otherwise.
    """
    result = pipe(data)
    if result and "label" in result[0]:
        return result[0]["label"] == "injection"
    return False  # Default to False if no valid result


def process_input(input_text, system_message=None):
    try:
        attack_detected = detect_prompt_injection_result([input_text])
        # This needs to return False if the attack was detected, it means the attack failed, so success = False
        return not attack_detected
    except Exception as e:
        # Re-raise the exception to be caught and logged by the tester
        raise e
