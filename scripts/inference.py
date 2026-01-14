#!/usr/bin/env python3
# Inference script for your fine-tuned model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
device = "cuda" if torch.cuda.is_available() else "cpu"
class Inference():
    def __init__(self, model_path="./my-finetuned-model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto"
        )

    def generate_response(self, instruction, max_tokens=200):
        prompt = f"### Instruction:\n{instruction}\n\n### Response:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the response part
        if "### Response:" in response:
            response = response.split("### Response:")[1].strip()
        return response

# Example usage
if __name__ == "__main__":
    print("Fine-tuned Model Inference")
    print("-" * 40)
    inference = Inference()
    
    while True:
        user_input = input("\nEnter instruction (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        response = inference.generate_response(user_input)
        print(f"\nResponse: {response}")
