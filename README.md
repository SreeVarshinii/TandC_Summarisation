# Legal Summarization with Explanatory Injection

## Project Overview
This project aims to enhance legal text summarization by not only shortening the content but also making it more accessible to laypeople. This is achieved by:
1.  **Abstractive Summarization**: Using advanced LLMs to generate concise summaries.
2.  **Explanatory Injection**: Automatically detecting complex legal terms in the summary and injecting definitions (inline or appended) to improve comprehension.

## Models & Architecture

### 1. Models Evaluated
*   **Google Flan-T5 Base (Fine-Tuned)**: The core model of this project. It was instruction-tuned on legal datasets to generate summaries with a formal yet accessible tone.
*   **Mistral 7B (Quantized)**: A powerful general-purpose LLM used for comparison and as a strong zero-shot baseline.
*   **BART (sshleifer/distilbart-cnn-12-6)**: A standard baseline for summarization tasks to benchmark performance.

### 2. Key Components
*   **`src/summarization.py`**: Handles loading models and generating summaries.
*   **`src/explanation.py`**: A dedicated module that identifies difficult terms in the generated summary and injects definitions/explanations.
*   **`src/compare_models.py`**: A script to run side-by-side comparisons of all models on a test dataset.
*   **`src/app.py`**: A Streamlit web application for interactive user testing.

## Steps Taken

### Phase 1: Data Preparation
*   Ingested legal datasets (Parquet format) containing legal text and reference summaries.
*   Performed data inspection using `scripts/inspect_data.py`.

### Phase 2: Model Development
*   **Fine-tuning**: Fine-tuned `google/flan-t5-base` using LoRA (Low-Rank Adaptation) for efficient training on legal instruction data.
*   **Integration**: Integrated Mistral 7B for high-quality generation capability.

### Phase 3: Evaluation & Comparison
*   Implemented `src/compare_models.py` to evaluate models using:
    *   **ROUGE Scores**: For n-gram overlap with reference summaries.
    *   **Readability Metrics**: Flesch-Kincaid Grade Level.
    *   **Explanation Count**: Measuring the frequency of injected explanations.
*   Generated `comparison_results.csv` containing detailed metrics for every sample.

### Phase 4: Visualization & Showcase
*   Created **`notebooks/showcase_results.ipynb`** to visualize the performance metrics (Bar charts for ROUGE, Readability).
*   Built a **Streamlit App** (`src/app.py`) allowing users to input raw legal text and see summaries from different models side-by-side.

## Results
The comparison results are stored in `comparison_results.csv`.
*   **Quantitative**: Check `notebooks/showcase_results.ipynb` for charts. Generally, Flan-T5 (Fine-tuned) shows improved domain adaptation over the base model, while Mistral provides high fluency.
*   **Qualitative**: The inclusion of the `ExplanationInjector` significantly improves the readability of summaries for non-experts by clarifying jargon.

## How to Run

### prerequisites
*   Python 3.8+
*   Install dependencies: `pip install -r requirements.txt`

### 1. Interactive Demo
Run the Streamlit app to try the models yourself:
```bash
streamlit run src/app.py
```

### 2. Reproduce Comparison
Run the comparison script to generate new results:
```bash
python src/compare_models.py --model_path "models/your_checkpoint"
```

### 3. Analyze Results
Open the Jupyter notebook:
```bash
jupyter notebook notebooks/showcase_results.ipynb
```
