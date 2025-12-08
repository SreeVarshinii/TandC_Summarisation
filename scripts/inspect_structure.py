import kagglehub
import os

def inspect_structure():
    path = kagglehub.dataset_download("kageneko/legal-case-document-summarization")
    print("Base Path:", path)
    
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        # Print first 5 files
        for f in files[:5]:
            print('{}{}'.format(subindent, f))
        if len(files) > 5:
            print(f'{subindent}... ({len(files)-5} more files)')

if __name__ == "__main__":
    inspect_structure()
