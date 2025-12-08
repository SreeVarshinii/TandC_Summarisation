import pandas as pd
import os

def inspect_parquet(filepath):
    print(f"--- Inspecting {filepath} ---")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    try:
        df = pd.read_parquet(filepath)
        print(f"Shape: {df.shape}")
        print("Columns:")
        print(df.columns.tolist())
        print("\nTypes:")
        print(df.dtypes)
        print("\nFirst 2 rows:")
        print(df.head(2).to_string()) 
        print("\n")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

if __name__ == "__main__":
    inspect_parquet("data/parquets/mistral_instruction_data.parquet")
    inspect_parquet("data/parquets/ledgar_glossary.parquet")
