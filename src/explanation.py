import json
import re
import os

class ExplanationInjector:
    def __init__(self, glossary_path="data/jsons/ledgar_glossary.json"):
        """
        Initializes the injector by loading the glossary.
        """
        self.glossary = {}
        if os.path.exists(glossary_path):
            with open(glossary_path, 'r', encoding='utf-8') as f:
                self.glossary = json.load(f)
            print(f"Loaded glossary with {len(self.glossary)} terms.")
        else:
            print(f"Warning: Glossary file not found at {glossary_path}")

    def inject(self, text):
        """
        Injects definitions for glossary terms found in the text.
        Replaces the first occurrence of each term with "term (definition)".
        """
        if not self.glossary:
            return text
            
        modified_text = text
        
        # Sort terms by length (descending) to avoid partial matches interfering (though regex \b helps)
        # However, for efficiency we iterate. A more optimized approach might use a single regex pass,
        # but for this scale, iteration is acceptable.
        sorted_terms = sorted(self.glossary.keys(), key=len, reverse=True)
        
        for term in sorted_terms:
            definition = self.glossary[term]
            
            # Simple regex for whole word match, case insensitive
            pattern = re.compile(rf'\b({re.escape(term)})\b', re.IGNORECASE)
            
            # Search if term exists
            match = pattern.search(modified_text)
            if match:
                # We only want to replace the FIRST occurrence
                # Construct the replacement string: "OriginalTerm (definition)"
                # We use a lambda to properly capture the original casing if we wanted, 
                # but 'match.group(1)' gives us the exact text found.
                
                # Check if it's already defined? e.g. "term (definition)" pattern?
                # For now, simplistic injection.
                
                def replace_func(m):
                    return f"{m.group(1)} ({definition})"
                
                # count=1 ensures only the first one is replaced
                modified_text = pattern.sub(replace_func, modified_text, count=1)
                
        return modified_text

if __name__ == "__main__":
    # Test Block
    injector = ExplanationInjector()
    
    test_sentences = [
        "This agreement may be executed in counterparts.",
        "The governing laws shall apply to all jurisdictions.",
        "No specific indemnifications are provided in this section.",
        "Compliance with laws is mandatory." # Testing multi-word term
    ]
    
    print("\n--- Injection Tests ---")
    for sent in test_sentences:
        print(f"Original: {sent}")
        print(f"Injected: {injector.inject(sent)}")
        print("-" * 30)
