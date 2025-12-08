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

class ModelComparator:
    def __init__(self, sample_size=5):
        self.sample_size = sample_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize Metrics
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize Our System Components
        print("Loading Flan-T5 (Our System)...")
        self.our_model = LegalSummarizer() # Defaults to google/flan-t5-base
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
            df = pd.read_parquet(data_path)
            # Use samples that have both input and output
            df = df.dropna(subset=['input', 'output']).sample(self.sample_size, random_state=42)
        except Exception as e:
            print(f"Error loading data: {e}")
            return

        results = []
        
        print(f"\n--- Running Comparison on {self.sample_size} samples ---")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            original_text = row['input']
            reference_summary = row['output']
            
            # 1. BART Baseline
            bart_summary = self.generate_bart(original_text)
            bart_metrics = self.calculate_metrics(bart_summary, reference_summary)
            
            # 2. Flan-T5 (No Explain)
            # Use 'Formal' tone and 'Detailed' length as standard comparison
            flan_summary = self.our_model.summarize(original_text, tone="Formal", length="Detailed")
            flan_metrics = self.calculate_metrics(flan_summary, reference_summary)
            
            # 3. Flan-T5 + Explanation (Our Proposed System)
            flan_explained = self.injector.inject(flan_summary)
            our_metrics = self.calculate_metrics(flan_explained, reference_summary)
            
            # Store Results
            results.append({
                "model": "BART",
                "id": idx,
                "summary": bart_summary,
                **bart_metrics
            })
            results.append({
                "model": "Flan-T5 (No Explain)",
                "id": idx,
                "summary": flan_summary,
                **flan_metrics
            })
            results.append({
                "model": "Flan-T5 + Explain",
                "id": idx,
                "summary": flan_explained,
                **our_metrics
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
    comparator = ModelComparator(sample_size=5) # Small sample for quick verification
    comparator.run_comparison()
