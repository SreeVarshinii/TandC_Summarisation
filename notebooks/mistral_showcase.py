#!/usr/bin/env python
# coding: utf-8

# # Mistral Legal Summarization Showcase
# 
# This notebook demonstrates the capabilities of the **Mistral 7B** model for summarising complex legal documents.

# In[ ]:


import sys
import os

# Ensure project root is in path for imports
# If running from notebooks folder, go up one level
if os.getcwd().endswith('notebooks'):
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
else:
    project_root = os.getcwd()

if project_root not in sys.path:
    sys.path.append(project_root)

from src.summarization import LegalSummarizer

# Initialize Summarizer with Local Mistral Model
# Using forward slashes to avoid Windows escape sequence warnings
model_path = "models/mistral_v0.3/content/TandC_Summarisation/models/mistral_legal_finetuned"
full_model_path = os.path.join(project_root, model_path)

if not os.path.exists(full_model_path):
    print(f"Model not found at {full_model_path}, falling back to base model or checking other paths.")
    # Fallback for demonstration if local unzip didn't match expectation, though we expect it there.
    model_path = "google/flan-t5-base" 
else:
    model_path = full_model_path

print(f"Loading model from: {model_path}")
summarizer = LegalSummarizer(model_name=model_path)


# ## 1. Input Legal Text

# In[ ]:


long_legal_text = """
TERMS OF SERVICE

1. ACCEPTANCE OF TERMS
By accessing this website, you agree to be bound by these Terms and Conditions of Use, all applicable laws and regulations, and agree that you are responsible for compliance with any applicable local laws. If you do not agree with any of these terms, you are prohibited from using or accessing this site.

2. USE LICENSE
Permission is granted to temporarily download one copy of the materials (information or software) on Company's website for personal, non-commercial transitory viewing only. This is the grant of a license, not a transfer of title, and under this license you may not:
   a. modify or copy the materials;
   b. use the materials for any commercial purpose, or for any public display (commercial or non-commercial);
   c. attempt to decompile or reverse engineer any software contained on Company's website;
   d. remove any copyright or other proprietary notations from the materials; or
   e. transfer the materials to another person or "mirror" the materials on any other server.

3. DISCLAIMER
The materials on Company's website are provided "as is". Company makes no warranties, expressed or implied, and hereby disclaims and negates all other warranties, including without limitation, implied warranties or conditions of merchantability, fitness for a particular purpose, or non-infringement of intellectual property or other violation of rights. Further, Company does not warrant or make any representations concerning the accuracy, likely results, or reliability of the use of the materials on its Internet web site or otherwise relating to such materials or on any sites linked to this site.

4. LIMITATIONS
In no event shall Company or its suppliers be liable for any damages (including, without limitation, damages for loss of data or profit, or due to business interruption,) arising out of the use or inability to use the materials on Company's Internet site, even if Company or a Company authorized representative has been notified orally or in writing of the possibility of such damage. Because some jurisdictions do not allow limitations on implied warranties, or limitations of liability for consequential or incidental damages, these limitations may not apply to you.
"""

print("Input Text Length:", len(long_legal_text))


# ## 2. Generate Summary (Formal / Detailed)

# In[ ]:


summary = summarizer.summarize(long_legal_text, tone="Formal", length="Detailed")
print("\n--- Generated Summary ---\n")
print(summary)


# ## 3. Explain Key Terms

# In[ ]:


from src.explanation import ExplanationInjector
injector = ExplanationInjector()

explained_summary = injector.inject(summary)
print("\n--- Explained Summary ---\n")
print(explained_summary)

