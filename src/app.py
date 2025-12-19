import streamlit as st
import sys
import os
import spacy

# Ensure src is in path
sys.path.append(os.getcwd())

from src.summarization import LegalSummarizer
from src.explanation import ExplanationInjector
from src.preprocessing import segment_clauses

# Page Config
st.set_page_config(page_title="LegalSummarizer AI", layout="wide")

# Initialize Resources (Cached)
@st.cache_resource
def load_resources():
    print("Loading models...")
    # Priority: Local Mistral > Local Flan-T5 > Base Flan-T5
    model_path = "google/flan-t5-base"
    
    # Path found from unzip structure
    deep_mistral_path = "models/mistral_v0.3/content/TandC_Summarisation/models/mistral_legal_finetuned"
    
    if os.path.exists(deep_mistral_path):
        model_path = deep_mistral_path
        print(f"Using local Mistral model at {model_path}")
    elif os.path.exists("models/flan-t5-kaggle-final"):
        model_path = "models/flan-t5-kaggle-final"
        print(f"Using local Flan-T5 model at {model_path}")
        
    try:
        summarizer = LegalSummarizer(model_name=model_path)
    except:
        st.error(f"Could not load model from {model_path}. Checking internet or path.")
        return None, None, None
        
    injector = ExplanationInjector()
    
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        
    return summarizer, injector, nlp

summarizer, injector, nlp = load_resources()

# Sidebar Controls
st.sidebar.title("Configuration")

tone = st.sidebar.selectbox("Select Tone", ["Formal", "Neutral", "Casual"])
length = st.sidebar.selectbox("Summarization Verbosity", ["Short", "Standard", "Detailed"])
focus = st.sidebar.multiselect("Focus Clause Filtering", ["Privacy", "Billing", "Liability"], default=[])

# Main Application
st.title("⚖️ Legal Document Summarizer")
st.markdown("Past a Terms & Conditions (T&C) or legal text below to generate an explained summary.")

input_text = st.text_area("Legal Text Input", height=300, value="Paste text here...")

if st.button("Generate Summary"):
    if not input_text or len(input_text) < 10:
        st.warning("Please enter valid text.")
    else:
        with st.spinner("Processing..."):
            # 1. Processing / Focus Filtering
            processed_text = input_text
            
            if focus:
                st.info(f"Filtering for clauses related to: {', '.join(focus)}")
                clauses = segment_clauses(input_text, nlp)
                
                # Define Keywords (Extending from preprocessing.py)
                keywords = {
                    "Privacy": ['privacy', 'confidential', 'data protection', 'personal information', 'disclosure', 'gdpr', 'cookies'],
                    "Billing": ['payment', 'fees', 'charges', 'billing', 'subscription', 'refund', 'credit card', 'invoice'],
                    "Liability": ['liability', 'liable', 'indemnify', 'indemnification', 'damages', 'harmless', 'warranty', 'disclaimer']
                }
                
                relevant_clauses = []
                for clause in clauses:
                    lower_c = clause.lower()
                    if any(ft in focus for ft in ["Privacy", "Billing", "Liability"]):
                        # Check selected topics
                        is_relevant = False
                        for topic in focus:
                            if any(kw in lower_c for kw in keywords.get(topic, [])):
                                is_relevant = True
                                break
                        if is_relevant:
                            relevant_clauses.append(clause)
                
                if relevant_clauses:
                    processed_text = " ".join(relevant_clauses)
                    st.success(f"Focused Text Segmented ({len(relevant_clauses)} clauses found).")
                    with st.expander("View Filtered Text"):
                        st.write(processed_text)
                else:
                    st.warning("No clauses matched your focus topics. Summarizing original text.")
            
            # 2. Summarization
            summary = summarizer.summarize(processed_text, tone=tone, length=length)
            
            # 3. Explanation Injection
            final_output = injector.inject(summary)
            
            # Display
            st.subheader("Summary")
            st.write(final_output)
            
            # Visualization of Explanations (Highlighting)
            # Simple heuristic: find "term (definition)" and highlight
            # For the demo, simply showing the text is good.
