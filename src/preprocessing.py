import pandas as pd
import spacy
import re
import os

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    return pd.read_parquet(filepath)

def setup_spacy():
    print("Loading SpaCy structure...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading en_core_web_sm...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

def segment_clauses(text, nlp):
    """
    Splits text into clauses using SpaCy and custom heuristics.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    clauses = []
    for sent in sentences:
        # Heuristic 1: Split on semi-colons
        parts = sent.split(';')
        for part in parts:
            # Heuristic 2: Split on numbered lists like (1), (a) if they start a new clause logic
            # For simplicity, we keep them as is but you could refine regex here
            # We filter out very short fragments
            clean_part = part.strip()
            if len(clean_part) > 10:  # arbitrary filter for noise
                clauses.append(clean_part)
                
    return clauses

def focus_filtering(clauses):
    """
    Filters clauses based on keywords.
    """
    privacy_keywords = ['privacy', 'confidential', 'data protection', 'personal information', 'disclosure', 'gdpr']
    liability_keywords = ['liability', 'liable', 'indemnify', 'indemnification', 'damages', 'harmless', 'warranty']
    
    filtered_data = []
    
    for clause in clauses:
        lower_clause = clause.lower()
        matched_topic = None
        
        # Check Privacy
        if any(kw in lower_clause for kw in privacy_keywords):
            matched_topic = "Privacy"
        # Check Liability (if not already Privacy, or could be both? Let's prioritize or allow multi-label.
        # Requirement says "Focus Filtering", let's assign one. If both, maybe "Liability" is more critical? 
        # Or just "Both". Let's update to return first match or specific precedence.
        # Let's check Liability as well. form list.
        
        topics = []
        if any(kw in lower_clause for kw in privacy_keywords):
            topics.append("Privacy")
        if any(kw in lower_clause for kw in liability_keywords):
            topics.append("Liability")
            
        if topics:
             filtered_data.append({
                 "clause_text": clause,
                 "matched_topic": ", ".join(topics) 
             })
             
    return filtered_data

def process_and_save(input_file, output_file):
    df = load_data(input_file)
    nlp = setup_spacy()
    
    all_filtered_clauses = []
    
    print("Processing documents...")
    for index, row in df.iterrows():
        text = row['input'] # Assuming 'input' column has the text
        if not isinstance(text, str):
            continue
            
        clauses = segment_clauses(text, nlp)
        filtered = focus_filtering(clauses)
        
        for item in filtered:
            item['original_index'] = index
            all_filtered_clauses.append(item)
            
    print(f"Extracted {len(all_filtered_clauses)} relevant clauses.")
    
    out_df = pd.DataFrame(all_filtered_clauses)
    
    # Reorder columns
    if not out_df.empty:
        out_df = out_df[['original_index', 'clause_text', 'matched_topic']]
        out_df.to_parquet(output_file)
        print(f"Saved to {output_file}")
    else:
        print("No clauses matched. Nothing saved.")

if __name__ == "__main__":
    input_path = "data/parquets/mistral_instruction_data.parquet"
    output_path = "data/preprocessed/filtered_clauses.parquet"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    process_and_save(input_path, output_path)
