import kagglehub
import os
import pandas as pd
import json
import re
from tqdm import tqdm

def load_glossary(path="data/jsons/ledgar_glossary.json"):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_terms(text, glossary):
    found_terms = []
    text_lower = text.lower()
    for term, definition in glossary.items():
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_terms.append((term, definition))
    return found_terms

def process_kaggle_data():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("kageneko/legal-case-document-summarization")
    print(f"Dataset Path: {path}")

    glossary = load_glossary()
    data = []

    # Traverse directory to find 'judgement' and 'summary' folders
    print("Traversing files...")
    for root, dirs, files in os.walk(path):
        if 'judgement' in os.path.basename(root):
            # Potential data folder. Look for sibling 'summary' folder
            # Usually structure is parent/judgement and parent/summary
            parent = os.path.dirname(root)
            summary_dir = os.path.join(parent, 'summary')
            
            if not os.path.exists(summary_dir):
                continue
                
            # Determine split from path (heuristic)
            split = 'train'
            if 'test' in parent.lower(): split = 'test'
            elif 'dev' in parent.lower() or 'val' in parent.lower(): split = 'validation'
            
            # Use 'region' from path? e.g. /UK/train-data/judgement
            # Not strictly needed but helpful.
            
            for file in files:
                if not file.endswith('.txt'): continue
                
                judgement_path = os.path.join(root, file)
                summary_path = os.path.join(summary_dir, file)
                
                if os.path.exists(summary_path):
                    with open(judgement_path, 'r', encoding='utf-8', errors='ignore') as f:
                        input_text = f.read().strip()
                    with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
                        output_text = f.read().strip()
                        
                    if input_text and output_text:
                        data.append({
                            'input': input_text,
                            'original_summary': output_text,
                            'split': split,
                            'filename': file
                        })
    
    print(f"Found {len(data)} document pairs. Processing augmentations...")
    
    processed_rows = []
    
    for item in tqdm(data):
        input_text = item['input']
        original_summary = item['original_summary']
        
        # Augmentation Logic
        instruction = f"Summarize this legal case and explain any terms. Text: {input_text[:20000]}" 
        # Truncating input in instruction text is just for the prompt string, 
        # actual input to model will be tokenized and truncated.
        # But we pass the whole text. 
        # Note: T5 has limit. We'll leave truncation to tokenizer in train.py, or truncate here to be safe?
        # Let's keep it raw here mainly.
        
        # Find terms in INPUT (Judgement)
        # Limiting search to first 10k chars for speed/relevance? 
        # Or search summary? 
        # Preparing Explanation based on Input is better for "educational" value during reading.
        # But for "Explanation of generated summary", we should check summary.
        # The previous plan said "Scan INPUT". Let's stick to that but optimized.
        terms = find_terms(input_text[:50000], glossary) # Limit search space for performance
        
        explanation_section = ""
        if terms:
            # Deduplicate by key
            seen = set()
            unique_terms = []
            for t, d in terms:
                if t not in seen:
                    unique_terms.append((t, d))
                    seen.add(t)
            
            lines = [f"- {t}: {d}" for t, d in unique_terms[:10]] # Limit to top 10 to avoid huge outputs
            explanation_section = "\n\nExplanation:\n" + "\n".join(lines)
        else:
             explanation_section = "\n\nExplanation:\n- No specific legal terms found."
             
        target = f"Summary:\n{original_summary}{explanation_section}"
        
        processed_rows.append({
            'input': instruction,
            'output': target,
            'split': item['split'],
            'original_id': item['filename']
        })
        
    df = pd.DataFrame(processed_rows)
    
    output_path = "data/training/kaggle_legal_augmented.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    print(f"Saved {len(df)} rows to {output_path}")

if __name__ == "__main__":
    process_kaggle_data()
