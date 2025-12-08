import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
import textstat
import re
from src.summarization import LegalSummarizer
from src.explanation import ExplanationInjector
from tqdm import tqdm
import argparse

class ModelComparator:
    def __init__(self, model_path="google/flan-t5-base", sample_size=5):
        self.sample_size = sample_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize Metrics
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize Our System Components (Fine-Tuned or Base)
        print(f"Loading Evaluated Model: {model_path}...")
        try:
             self.our_model = LegalSummarizer(model_name=model_path)
             self.model_label = "Fine-Tuned Flan-T5" if "models" in model_path else "Flan-T5 Base"
        except Exception as e:
             print(f"Error loading model: {e}")
             sys.exit(1)

        self.injector = ExplanationInjector()
        
        # Initialize Baseline (BART)
        print("Loading BART (Baseline)...")
        self.bart_name = "sshleifer/distilbart-cnn-12-6"
        self.bart_tokenizer = AutoTokenizer.from_pretrained(self.bart_name)
        self.bart_model = AutoModelForSeq2SeqLM.from_pretrained(self.bart_name).to(self.device)
        
    def generate_bart(self, text):
        inputs = self.bart_tokenizer([text], max_length=1024, return_tensors="pt", truncation=True).to(self.device)
        summary_ids = self.bart_model.generate(inputs["input_ids"], max_new_tokens=150, num_beams=4, early_stopping=True)
        return self.bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def calculate_metrics(self, generated, reference):
        # ROUGE
        scores = self.scorer.score(reference, generated)
        rouge1 = scores['rouge1'].fmeasure
        rougeL = scores['rougeL'].fmeasure
        
        # Readability
        grade = textstat.flesch_kincaid_grade(generated)
        
        # Explanation Count (Regex for 'term (definition)')
        # Heuristic: word followed by parenthesized phrase
        # Note: This is an approximation. 
        explanation_count = len(re.findall(r'\b\w+\s\([^)]+\)', generated))
        
        return {
            "rouge1": round(rouge1, 4),
            "rougeL": round(rougeL, 4),
            "grade_level": grade,
            "explanation_count": explanation_count
        }

    def run_comparison(self, data_path="data/parquets/mistral_instruction_data.parquet"):
        try:
            print(f"Loading data from {data_path}...")
            df = pd.read_parquet(data_path)
            # Use samples that have both input and output
            # If using augmented data, columns might be different, handle flexibly
            if 'original_summary' in df.columns:
                 # Prefer original summary as reference if available
                 df['output'] = df['original_summary']
            
            df = df.dropna(subset=['input', 'output']).sample(self.sample_size, random_state=42)
        except Exception as e:
            print(f"Error loading data: {e}")
            return

        results = []
        
        print(f"\n--- Running Comparison on {self.sample_size} samples ---")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            original_text = row['input']
            # If input has instruction prefix "Summary etc. Text: ", we might want to strip it for BART?
            # Or just pass raw. BART is robust enough.
            # But for Flan-T5 fine-tuned, we trained on "Summarize... Text: ..." so keep it?
            # Wait, src/summarization.py constructs prompt again! 
            # If input ALREADY has prompt, we shouldn't add it again.
            # But 'input' column in augmented parquet DOES have it.
            # Let's trust 'input' meant for model contains just text if it's original data, 
            # OR contains prompt if it's processed.
            # Let's assume input is just Text for generalized comparison.
            # If using augmented, input IS prompt. 
            
            reference_summary = row['output']
            
            # Helper to strip prompt if needed for BART
            just_text = original_text
            if "Text: " in original_text:
                 just_text = original_text.split("Text: ")[-1]

            # 1. BART Baseline (Pure Extraction/Abstractive Baseline)
            bart_summary = self.generate_bart(just_text[:1024]) # Truncate for BART
            bart_metrics = self.calculate_metrics(bart_summary, reference_summary)
            
            # 2. Evaluated Model (Your Trained Model)
            # Pass original_text (if it's raw text, summarizer adds prompt. If it's prompt, summarizer adds prompt again??)
            # src/summarization.py `construct_prompt` wraps text.
            # If we pass already wrapped text, we get double wrap.
            # So pass `just_text`.
            
            flan_summary = self.our_model.summarize(just_text, tone="Formal", length="Detailed")
            flan_metrics = self.calculate_metrics(flan_summary, reference_summary)
            
            # 3. Post-Hoc Explanation (Optional check if fine-tuning learned it vs injection)
            # The 'Fine-Tuned' model might generate explanations naturally.
            # We can check that by counting explanations in `flan_summary`.
            
            # Also run strict Injection on top?
            flan_injected = self.injector.inject(flan_summary)
            injected_metrics = self.calculate_metrics(flan_injected, reference_summary)
            
            # Store Results
            results.append({
                "model": "BART (Baseline)",
                "id": idx,
                "summary": bart_summary,
                **bart_metrics
            })
            results.append({
                "model": self.model_label,
                "id": idx,
                "summary": flan_summary,
                **flan_metrics
            })
            results.append({
                "model": f"{self.model_label} + Explicit Injection",
                "id": idx,
                "summary": flan_injected,
                **injected_metrics
            })
            
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Aggregated Metrics
        print("\n--- Aggregated Results ---")
        agg = results_df.groupby("model")[['rouge1', 'rougeL', 'grade_level', 'explanation_count']].mean()
        print(agg)
        
        # Save detailed results
        results_df.to_csv("comparison_results.csv", index=False)
        print("\nDetailed results saved to comparison_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Models for Legal Summarization")
    parser.add_argument("--model_path", type=str, default="google/flan-t5-base", help="Path to your fine-tuned model checkpoint")
    parser.add_argument("--data_path", type=str, default="data/parquets/mistral_instruction_data.parquet", help="Path to test data parquet")
    
    args = parser.parse_args()
    
    comparator = ModelComparator(model_path=args.model_path, sample_size=5)
    comparator.run_comparison(data_path=args.data_path)
