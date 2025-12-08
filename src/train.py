import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import os
os.environ["WANDB_DISABLED"] = "true"

def train():
    model_name = "google/flan-t5-base"
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Load Augmented Data
    data_path = "data/training/augmented_train.parquet"
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    train_df = df 
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(train_df)
    
    def preprocess_function(examples):
        inputs = examples["input"]
        targets = examples["output"]
        
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
        labels = tokenizer(targets, max_length=1024, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    print("Tokenizing data...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    # Split
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./models/checkpoints",
        eval_strategy="steps", # Changed to steps for quick check
        eval_steps=2,
        learning_rate=2e-5,
        per_device_train_batch_size=1, # Reduced to 1
        per_device_eval_batch_size=1,  # Reduced to 1
        gradient_accumulation_steps=4, # Simulate larger batch
        weight_decay=0.01,
        save_total_limit=2,
        max_steps=5, # Short run for verification
        predict_with_generate=True,
        logging_steps=1,
        use_cpu=not torch.cuda.is_available() 
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    output_dir = "models/flan-t5-legal-explained"
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train()
