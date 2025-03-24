import os
from typing import Dict, Any, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def process_input(input_text: str, system_message: Optional[str] = None) -> str:
    """
    Process the input prompt before sending it to the model
    
    Args:
        input_text (str): The main prompt or text to be processed
        system_message (str, optional): A system or meta-prompt, if applicable
        
    Returns:
        str: The model's response
    """
    # Initialize model and tokenizer
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HUGGINGFACE_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(model_name, token=os.getenv("HUGGINGFACE_TOKEN"))
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Combine system message and input text if provided
    full_prompt = f"{system_message}\n{input_text}" if system_message else input_text
    
    # Tokenize input
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]
    
    # Generate text with dynamic max_length
    max_new_tokens = 100  # Number of new tokens to generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

class HuggingFaceGPT2Target:
    def __init__(self):
        self.model_name = "openai-community/gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=os.getenv("HUGGINGFACE_TOKEN"))
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, token=os.getenv("HUGGINGFACE_TOKEN"))
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text from the model given a prompt
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode and return the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def __call__(self, prompt: str) -> Dict[str, Any]:
        """
        Main interface for the target
        """
        try:
            response = self.generate(prompt)
            return {
                "success": True,
                "response": response
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 