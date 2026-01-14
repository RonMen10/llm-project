from huggingface_hub import login
from huggingFace_token import token
login(token=token)

#!/usr/bin/env python3

import torch
import os
import json
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def main():
    print("=" * 60)
    print("GUARANTEED WORKING LLM Fine-tuning Pipeline")
    print("=" * 60)
    
    # ========== CONFIGURATION ==========
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Public, no auth needed
    OUTPUT_DIR = "./my-finetuned-model"
    
    # ========== 1. LOAD TOKENIZER ==========
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ========== 2. CREATE SIMPLE DATASET ==========
    print("\n2. Creating dataset...")
    
    # Simple instruction-response pairs
    with open('data/training_data.json', 'r') as f:
        training_examples = json.load(f)


    
    # Format for training
    def format_instruction(example):
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
        return {"text": text}
    
    # Create dataset
    dataset = Dataset.from_list(training_examples)
    dataset = dataset.map(format_instruction)
    print(f"Created dataset with {len(dataset)} examples")
    
    # ========== 3. LOAD MODEL WITH 4-BIT QUANTIZATION ==========
    print("\n3. Loading model with 4-bit quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # ========== 4. CONFIGURE LoRA ==========
    print("\n4. Configuring LoRA...")
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}% of total)")
    
    # ========== 5. SETUP TRAINING ==========
    print("\n5. Setting up training...")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=300,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        logging_steps=10,
        # save_strategy="epoch",
        save_strategy="steps",         # Save by steps (not epochs)
        save_steps=50,                 # Save every 50 steps
        save_total_limit=3,
        learning_rate=1e-4,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        push_to_hub=False,
        remove_unused_columns=False,
        fp16=False,
        bf16=False
    )
    
    # ========== 6. FORMATTING FUNCTION ==========
    print("\n6. Setting up formatting function...")
    
    # This is the KEY FIX for latest SFTTrainer
    def formatting_func(example):
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
        return text
    
    # Apply formatting to all examples
    formatted_texts = [formatting_func(ex) for ex in training_examples]
    
    # Create new dataset with formatted texts
    train_dataset = Dataset.from_dict({"text": formatted_texts})
    
    # ========== 7. INITIALIZE SFTTrainer ==========
    print("\n7. Initializing trainer...")
    
    # LATEST CORRECT API (March 2024+)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        #packing=False,
        # dataset_text_field="text",  # Field containing the text
    )
    
    # Note: tokenizer is NOT passed to SFTTrainer anymore in latest versions
    # It's automatically taken from the model
    
    # ========== 8. TRAIN ==========
    print("\n8. Starting training...")
    print("   This will take 5-10 minutes depending on your GPU...")
    
    trainer.train()
    
    # ========== 9. SAVE MODEL ==========
    print("\n9. Saving model...")
    
    # Save the model
    trainer.save_model()
    
    # Save tokenizer separately
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Model saved to: {OUTPUT_DIR}")
    
    # ========== 10. TEST THE MODEL ==========
    print("\n10. Testing the model...")
    
    # Load the model for testing
    from peft import AutoPeftModelForCausalLM
    
    test_model = AutoPeftModelForCausalLM.from_pretrained(
        OUTPUT_DIR,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Test with a prompt
    test_prompt = "### Instruction:\nWhat is Python?\n\n### Response:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = test_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "=" * 60)
    print("TEST OUTPUT:")
    print("=" * 60)
    print(response)
    print("=" * 60)
    
    # ========== 11. CREATE DEPLOYMENT FILES ==========
    print("\n11. Creating deployment files...")
    
    # Create a simple inference script
    inference_script = f"""#!/usr/bin/env python3
# Inference script for your fine-tuned model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "{OUTPUT_DIR}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    device_map="auto"
)

def generate_response(instruction, max_tokens=200):
    prompt = f"### Instruction:\\n{{instruction}}\\n\\n### Response:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response part
    if "### Response:" in response:
        response = response.split("### Response:")[1].strip()
    return response

# Example usage
if __name__ == "__main__":
    print("Fine-tuned Model Inference")
    print("-" * 40)
    
    while True:
        user_input = input("\\nEnter instruction (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        response = generate_response(user_input)
        print(f"\\nResponse: {{response}}")
"""
    
    with open(os.path.join(OUTPUT_DIR, "inference.py"), "w") as f:
        f.write(inference_script)
    
    # Create requirements.txt
    requirements = """torch>=2.1.0
transformers>=4.39.0
accelerate>=0.27.0
peft>=0.9.0
bitsandbytes>=0.42.0
"""
    
    with open(os.path.join(OUTPUT_DIR, "requirements.txt"), "w") as f:
        f.write(requirements)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Inference script: {OUTPUT_DIR}/inference.py")
    print(f"Requirements: {OUTPUT_DIR}/requirements.txt")
    print("\nTo use your model:")
    print(f"1. cd {OUTPUT_DIR}")
    print("2. pip install -r requirements.txt")
    print("3. python inference.py")
    print("=" * 60)

if __name__ == "__main__":
    main()