from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LegalSummarizer:
    def __init__(self, model_name="google/flan-t5-base"):
        print(f"Loading model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        print(f"Model loaded on {self.device}.")

    def construct_prompt(self, text, tone, length):
        """
        Constructs the prompt based on tone and length constraints.
        """
        prompt = f"Summarize the following legal text in plain English. Use {tone} language. Keep it {length}. Text: {text}"
        return prompt

    def summarize(self, text, tone="Formal", length="Detailed"):
        """
        Generates a summary for the given text.
        """
        prompt = self.construct_prompt(text, tone, length)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        
        # Adjust generation parameters if needed based on length, but prompt engineering usually handles it for Flan-T5
        # We can set max_new_tokens to allow for "Detailed" output
        max_tokens = 150 if length == "Detailed" else 50
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            num_beams=4, 
            early_stopping=True
        )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

if __name__ == "__main__":
    # Test Block
    summarizer = LegalSummarizer()
    
    test_text = """
    SECTION 1. LIABILITY OF BUSINESS ENTITIES PROVIDING USE OF FACILITIES TO NONPROFIT ORGANIZATIONS.
    (a) Definitions.--In this section:
    (1) Business entity.--The term ``business entity'' means a firm, corporation, association, partnership, consortium, joint venture, or other form of enterprise.
    (2) Facility.--The term ``facility'' means any real property, including any building, improvement, or appurtenance.
    """
    
    print("\n--- Test Case 1: Formal / Detailed ---")
    print(summarizer.summarize(test_text, tone="Formal", length="Detailed"))
    
    print("\n--- Test Case 2: Casual / Short ---")
    print(summarizer.summarize(test_text, tone="Casual", length="Short"))
