import pandas as pd
import json
import re
import os
from tqdm import tqdm

def load_glossary(path="data/jsons/ledgar_glossary.json"):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_terms(text, glossary):
    found_terms = []
    # Sort by length to prioritize longer matches if needed, though here we just want a list
    # Using set to avoid duplicates in the explanation list
    text_lower = text.lower()
    
    for term, definition in glossary.items():
        # Simple word boundary check
        # Escaping term for regex safety
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_terms.append((term, definition))
            
    return found_terms

def construct_augmented_data(input_file, output_file, glossary_path):
    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    glossary = load_glossary(glossary_path)
    
    augmented_rows = []
    
    print("Augmenting data...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        original_input = row['input']
        original_summary = row['output']
        
        if not isinstance(original_input, str) or not isinstance(original_summary, str):
            continue
            
        # 1. Update Instruction
        new_instruction = f"Rewrite the bill in plain language AND explain any legal terms in bullet points. Text: {original_input}"
        
        # 2. Find Terms for Explanation
        terms = find_terms(original_input, glossary)
        
        # 3. Construct Target
        explanation_section = ""
        if terms:
            explanation_lines = [f"- {term}: {defn}" for term, defn in terms]
            explanation_body = "\n".join(explanation_lines)
            explanation_section = f"\n\nExplanation:\n{explanation_body}"
        else:
            explanation_section = "\n\nExplanation:\n- No specific legal terms found."
            
        new_target = f"Summary:\n{original_summary}{explanation_section}"
        
        augmented_rows.append({
            "instruction": "custom_legal_explanation", # Tagging just in case
            "input": new_instruction,
            "output": new_target,
            "original_input": original_input # Keeping for reference
        })
        
    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df.to_parquet(output_file)
    print(f"Augmented data saved to {output_file} ({len(augmented_df)} rows)")
    
    # Preview
    print("\n--- Sample Augmentation ---")
    print(augmented_df.iloc[0]['output'])

if __name__ == "__main__":
    input_path = "data/parquets/mistral_instruction_data.parquet"
    output_path = "data/training/augmented_train.parquet"
    glossary_path = "data/jsons/ledgar_glossary.json"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    construct_augmented_data(input_path, output_path, glossary_path)
