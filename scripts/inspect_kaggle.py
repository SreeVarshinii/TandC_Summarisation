import kagglehub
import os
import pandas as pd

def inspect():
    print("Downloading dataset...")
    # Download latest version
    path = kagglehub.dataset_download("kageneko/legal-case-document-summarization")
    print("Path to dataset files:", path)
    
    # List files
    files = []
    for root, dirs, filenames in os.walk(path):
        for f in filenames:
            full_path = os.path.join(root, f)
            print(f"Found file: {f}")
            files.append(full_path)
            
    # Inspect first file (assuming csv or json)
    if files:
        first_file = files[0]
        print(f"\n--- Inspecting {os.path.basename(first_file)} ---")
        try:
            if first_file.endswith('.csv'):
                df = pd.read_csv(first_file)
            elif first_file.endswith('.json') or first_file.endswith('.jsonl'):
                df = pd.read_json(first_file, lines=True if first_file.endswith('.jsonl') else False)
            else:
                 # Try parquet
                 try:
                    df = pd.read_parquet(first_file)
                 except:
                    print("Unknown format.")
                    return

            print("Columns:", df.columns.tolist())
            print(df.head(2))
        except Exception as e:
            print(f"Error reading file: {e}")

if __name__ == "__main__":
    inspect()
