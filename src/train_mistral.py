import torch
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import pandas as pd
import os
import sys

os.environ["WANDB_DISABLED"] = "true"

def train_mistral(model_name, data_path, output_dir, max_steps):
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix for fp16 training

    print("Loading data...")
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Check columns
    if 'input' not in df.columns or 'output' not in df.columns:
        print("Data must have 'input' and 'output' columns.")
        return

    # Prepare data for Causal LM (Chat Format)
    # We combine Input + Output into one string for the model to learn to generate Output given Input.
    # Format: <s>[INST] {Instruction} [/INST] {Response}</s>
    
    print("Formatting data for Causal LM...")
    def format_chat(row):
        # Assuming 'input' column contains the instruction
        instruction = row['input']
        response = row['output']
        
        text = f"<s>[INST] {instruction} [/INST] {response}</s>"
        return text

    df['text'] = df.apply(format_chat, axis=1)
    dataset = Dataset.from_pandas(df[['text']])

    def preprocess(examples):
        return tokenizer(examples['text'], truncation=True, max_length=2048) # Mistral supports longer context

    print("Tokenizing...")
    tokenized_ds = dataset.map(preprocess, batched=True)
    
    # Split
    tokenized_ds = tokenized_ds.train_test_split(test_size=0.05) # Small validation set

    print(f"Loading model (4-bit): {model_name}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.config.use_cache = False # Silence warnings during training
    model.config.pretraining_tp = 1
    
    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, peft_config)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=max_steps, # Or num_train_epochs
        optim="paged_adamw_32bit",
        save_steps=100,
        fp16=True,
        # group_by_length=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['test'],
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2") # Use v0.3 if available/supported
    parser.add_argument("--data_path", type=str, default="data/training/kaggle_legal_augmented.parquet")
    parser.add_argument("--output_dir", type=str, default="models/mistral_legal_finetuned")
    parser.add_argument("--max_steps", type=int, default=500)
    args = parser.parse_args()

    train_mistral(args.model_name, args.data_path, args.output_dir, args.max_steps)
