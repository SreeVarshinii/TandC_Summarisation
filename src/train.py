import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import os
import argparse

os.environ["WANDB_DISABLED"] = "true"

def train(model_name, data_path, output_dir):
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    train_df = df 
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(train_df)
    
    def preprocess_function(examples):
        inputs = examples["input"]
        targets = examples["output"]
        
        # T5 uses a max length. For legal cases, 1024 might be too short for full context,
        # but T5-base has a limit. 
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
        labels = tokenizer(targets, max_length=1024, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    print("Tokenizing data...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    # Split - check if 'split' column exists from Kaggle processing, otherwise random split
    if 'split' in df.columns:
        print("Using existing splits...")
        train_data = tokenized_dataset.filter(lambda x: x['split'] == 'train')
        eval_data = tokenized_dataset.filter(lambda x: x['split'] == 'test' or x['split'] == 'validation')
        if len(eval_data) == 0:
             # Fallback if no test/val
             print("No test/validation split found, creating random split.")
             split_ds = tokenized_dataset.train_test_split(test_size=0.1)
             train_data = split_ds['train']
             eval_data = split_ds['test']
    else:
        print("Creating random train/test split...")
        split_ds = tokenized_dataset.train_test_split(test_size=0.1)
        train_data = split_ds['train']
        eval_data = split_ds['test']
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoint_dir,
        eval_strategy="steps", 
        eval_steps=500, # More reasonable for larger datasets
        save_steps=1000,
        learning_rate=2e-5,
        per_device_train_batch_size=2, 
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3, # Standard
        predict_with_generate=True,
        logging_steps=50,
        use_cpu=not torch.cuda.is_available() 
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print(f"Starting training on {len(train_data)} samples...")
    trainer.train()
    
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Flan-T5 for Legal Summarization")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base", help="Model checkpoint to load")
    parser.add_argument("--data_path", type=str, default="data/training/kaggle_legal_augmented.parquet", help="Path to training parquet file")
    parser.add_argument("--output_dir", type=str, default="models/flan-t5-legal-explained", help="Directory to save the fine-tuned model")
    
    args = parser.parse_args()
    
    train(args.model_name, args.data_path, args.output_dir)
