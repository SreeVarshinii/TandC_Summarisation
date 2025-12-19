from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig
import torch
import argparse
import sys
import os

class LegalSummarizer:
    def __init__(self, model_name="google/flan-t5-base"):
        print(f"Loading model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.is_causal = False
        
        try:
            # Check for Peft Adapter
            is_peft = False
            if os.path.exists(os.path.join(model_name, "adapter_config.json")):
                print("Detected Peft Adapter.")
                from peft import PeftModel, PeftConfig
                is_peft = True
                
                peft_config = PeftConfig.from_pretrained(model_name)
                base_model_name = peft_config.base_model_name_or_path
                print(f"Base model identified: {base_model_name}")
                
                # Load Base Model (Assuming Causal for Mistral/Llama usually)
                # We can check base model config
                # But typically Peft is used with Causal LMs in this context.
                # Let's try AutoConfig on base model to be sure.
                config = AutoConfig.from_pretrained(base_model_name)
                
            else:
                 config = AutoConfig.from_pretrained(model_name)

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Architecture Detection (Robust)
            archs = getattr(config, "architectures", [])
            if not archs:
                # Fallback or assume Seq2Seq if not clear, but print warning
                print("Warning: Could not detect architecture from config. Assuming based on behavior matches.")
                arch_str = ""
            else:
                arch_str = archs[0]

            if is_peft or any(arch in arch_str for arch in ["Mistral", "Llama", "CausalLM", "GPT"]):
                print(f"Detected Causal LM architecture.")
                self.is_causal = True
                
                # Load Model
                if is_peft:
                    # Load Base
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                    # Load Adapter
                    self.model = PeftModel.from_pretrained(base_model, model_name)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
            else:
                # Default to Seq2Seq
                print(f"Detected Seq2Seq architecture.")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
                
            print(f"Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model from {model_name}: {e}")
            sys.exit(1)

    def construct_prompt(self, text, tone, length):
        """
        Constructs the prompt based on tone and length constraints.
        """
        # Map Verbosity to Explicit Instructions
        length_instruction = ""
        if length == "Short": 
            length_instruction = "in exactly 3 sentences"
        elif length == "Standard": 
            length_instruction = "in about 5-6 sentences"
        elif length == "Detailed": 
            length_instruction = "in more than 10 sentences with full details"
        else:
            length_instruction = length

        instruction = f"Summarize the following legal text in plain English. Use {tone} language. Summarize {length_instruction}. Text: {text}"
        
        if self.is_causal:
            # Chat format for Causal LM
            # Input: <s>[INST] {Instruction} [/INST]
            return f"<s>[INST] {instruction} [/INST]"
        else:
            # Seq2Seq format
            return instruction

    def summarize(self, text, tone="Formal", length="Detailed"):
        """
        Generates a summary for the given text.
        """
        # Checks if text already has the prompt prefix (simple heuristic)
        if text.strip().startswith("Summarize the following legal text"):
             # If passed raw prompt, we might need to wrap it for Causal still?
             # Let's assume 'text' passed here is usually just the legal text.
             # If it IS the prompt, we trust user knows what they are doing only if seq2seq.
             if self.is_causal:
                 prompt = f"<s>[INST] {text} [/INST]" # Wrap raw prompt validly
             else:
                 prompt = text
        else:
             prompt = self.construct_prompt(text, tone, length)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048 if self.is_causal else 1024, truncation=True).to(self.device if not self.is_causal else self.model.device)
        
        # Adjust generation parameters based on length instructions
        if length == "Short":
            max_tokens = 75
            min_tokens = 10
        elif length == "Standard":
            max_tokens = 200
            min_tokens = 50
        elif length == "Detailed":
            max_tokens = 500
            min_tokens = 200
        else:
            max_tokens = 150
            min_tokens = 20
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            num_beams=4, 
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id
            # do_sample=True if self.is_causal else False, # Optional: sampling for variety in Mistral
        )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-processing for Causal LM (remove prompt)
        if self.is_causal:
            # The model might echo the prompt. Mistral Instruct usually doesn't if formatted right, but let's be safe.
            # Decoder-only often outputs [Prompt] [Completion].
            # We check if prompt is in summary and strip it.
            # More robust: use only new tokens. But encode/decode loop makes that hard to index exactly.
            # Heuristic: split by [/INST]
            if "[/INST]" in summary:
                summary = summary.split("[/INST]")[-1].strip()
                
        return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Legal Summarizer from Checkpoint")
    parser.add_argument("--model_path", type=str, default="google/flan-t5-base", help="Path to model checkpoint or HuggingFace model name")
    parser.add_argument("--text", type=str, help="Text to summarize (optional)")
    
    args = parser.parse_args()
    
    summarizer = LegalSummarizer(model_name=args.model_path)
    
    if args.text:
        print("\n--- Summary ---")
        print(summarizer.summarize(args.text))
    else:
        # Default Test Block
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
