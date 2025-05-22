# Document Analysis Pro 📄✨

**AI-powered document analysis for similarity detection, summarization, and keyword extraction, built with IBM's Granite LLM and Gradio.**

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [How It Works](#how-it-works)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Screenshots (Example Placeholders)](#screenshots-example-placeholders)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Overview

Document Analysis Pro is an intelligent application designed to help users quickly understand and analyze text-based documents. Leveraging the power of IBM's Granite 3.3 2B Large Language Model, it provides key insights through:

1.  **Document Similarity:** Compare two documents and get a similarity score.
2.  **Text Summarization:** Generate concise summaries of lengthy texts.
3.  **Keyword Extraction:** Identify the most important terms and concepts.

The application features a user-friendly web interface built with Gradio, making it accessible and easy to use for processing `.txt` and `.pdf` files. It's designed with performance and memory optimization in mind, including fallback mechanisms to ensure reliability.

## Features

*   **🧠 AI-Powered Analysis:** Utilizes IBM Granite 3.3 2B LLM for sophisticated NLP tasks.
*   **📊 Document Similarity Checker:**
    *   Upload two documents (TXT or PDF).
    *   Get a percentage score representing their content similarity.
    *   View document previews (approx. 20 sentences).
*   **📝 Document Summarizer:**
    *   Upload a document (TXT or PDF).
    *   Receive an AI-generated concise summary.
    *   Preview the original document (approx. 20 sentences).
*   **🔍 Keyword Extractor:**
    *   Upload a document (TXT or PDF).
    *   Get a list of key terms and concepts.
    *   Preview the original document (approx. 20 sentences).
*   **📄 File Support:** Handles `.txt` and `.pdf` file formats.
*   **⚙️ Optimized Performance & Memory Management:**
    *   Efficient text extraction from PDFs using `PyPDF2`.
    *   NLTK for text preprocessing (tokenization, stopword removal).
    *   Optimized model loading for CPU (with 8-bit quantization via `bitsandbytes`) and CUDA (with `bfloat16`).
    *   Uses `accelerate` for better device loading.
    *   Aggressive memory management with `gc.collect()` and `torch.cuda.empty_cache()`.
*   **🛡️ Fallback Mechanisms:** Implements simpler, non-AI methods (e.g., Jaccard similarity, extractive summarization) for analysis if the LLM encounters issues, ensuring application resilience.
*   **🎨 User-Friendly Interface:** Interactive UI built with Gradio, featuring custom CSS for an enhanced look and feel, progress indicators, and clear result displays.

## Technology Stack

*   **Language:** Python
*   **Large Language Model:** IBM Granite 3.3 2B (`ibm-granite/granite-3.3-2b-instruct`)
*   **Core AI/ML Libraries:**
    *   `transformers` (Hugging Face)
    *   `torch` (PyTorch)
    *   `bitsandbytes>=0.39.0` (for 8-bit quantization)
    *   `accelerate>=0.20.0` (for efficient model loading)
*   **NLP & Text Processing:**
    *   `nltk` (Natural Language Toolkit)
    *   `PyPDF2` (for PDF text extraction)
*   **Web Interface:** `gradio`
*   **Development Environment:** Designed to run from a Python script (e.g., `app.py`), potentially in environments like Google Colab or a local setup.

## How It Works

1.  **Model Initialization:** The IBM Granite model and tokenizer are loaded, with optimizations based on available hardware (CUDA or CPU).
2.  **File Upload:** The user uploads one or two documents through the Gradio interface.
3.  **Text Extraction:** Text is extracted from the uploaded files. PDFs are processed using `PyPDF2`.
4.  **Preprocessing (for Fallbacks):** For fallback mechanisms, text is cleaned (lowercase, remove special characters, numbers), tokenized, and stopwords are removed using NLTK.
5.  **LLM Interaction:**
    *   Appropriate prompts are constructed for the selected task (similarity, summarization, or keyword extraction), incorporating the extracted text (limited to a certain character count for performance).
    *   The prompt is sent to the loaded IBM Granite model using its chat template.
    *   The model generates a response based on parameters like `max_new_tokens`, `temperature`, `top_p`, and `top_k`.
6.  **Fallback (if needed):** If the LLM fails, times out, or returns a very short/invalid response:
    *   **Similarity:** Jaccard similarity on preprocessed tokens.
    *   **Summarization:** Extractive summary (first few sentences).
    *   **Keywords:** Frequency-based keyword extraction from preprocessed text.
7.  **Display Results:** The analysis results, document previews, and processing time are displayed to the user in the Gradio interface. Memory is cleared after each operation.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/document-analysis-pro.git
    cd document-analysis-pro
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The `app.py` script attempts to install packages at runtime using `subprocess`. However, for a more robust setup, you can pre-install them.

    *Create a `requirements.txt` file with the following content (or ensure these are installed):*
    ```txt
    gradio
    nltk
    transformers
    PyPDF2
    bitsandbytes>=0.39.0
    accelerate>=0.20.0
    torch
    ```
    *Then install using pip:*
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The script will still attempt to install them if it doesn't find them, which is useful for environments like Colab but can be redundant locally if you pre-install.)*

4.  **Download NLTK resources:**
    The script handles this automatically on the first run if resources are missing. You can also run this Python snippet once:
    ```python
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
    ```

5.  **Hugging Face Hub Login (Optional but Recommended):**
    To ensure smooth model downloading from Hugging Face Hub, especially if you encounter rate limits or access private models, log in using the Hugging Face CLI:
    ```bash
    pip install huggingface_hub
    huggingface-cli login
    ```
    (You'll need a Hugging Face account and an access token with at least read permissions.)

## Usage

Run the main Python script (`app.py`):

```bash
python app.py
```
This will:
1.  Attempt to install any missing required packages.
2.  Download NLTK resources if needed.
3.  Initialize and load the language model (this may take some time on the first run or if the model isn't cached).
4.  Launch the Gradio web application.

A local URL (e.g., `http://127.0.0.1:7860` or similar) will be printed to your console. Open this URL in your web browser. If `share=True` is used in `app.launch()`, a temporary public Gradio link will also be provided for sharing.

**Navigating the Application:**

1.  Choose a tab: "📊 Similarity Checker," "📝 Document Summarizer," or "🔍 Keyword Extractor."
2.  Use the file upload component(s) to select your `.txt` or `.pdf` document(s).
3.  Click the appropriate button (e.g., "Analyze Similarity," "Generate Summary").
4.  Wait for the processing to complete (progress will be shown).
5.  View the results, including any scores, generated text, and processing time.
6.  Expand the "Document Previews" accordion to see a snippet of the uploaded document(s).

## Screenshots (Example Placeholders)

**Similarity Checker Tab Interface:**
![Similarity Checker Tab](.screenshots/doc_similarity_checker_UI_1.png)
![Similarity Checker Tab](.screenshots/doc_similarity_checker_UI_2.png)
![Similarity Checker Tab](.screenshots/doc_similarity_checker_UI_3.png)
![Similarity Checker Tab](.screenshots/doc_similarity_checker_UI_4.png)

**Document Summarizer with Results:**
`![Document Summarizer Tab](https://via.placeholder.com/600x400.png?text=Summarizer+UI+with+Results)`

**Keyword Extractor Output:**
`![Keyword Extractor Tab](https://via.placeholder.com/600x400.png?text=Keyword+Extractor+UI)`

## Future Enhancements

*   [ ] Support for more file formats (e.g., `.docx` using `python-docx`, `.pptx`).
*   [ ] UI controls for LLM parameters (e.g., temperature, `max_new_tokens`).
*   [ ] Batch processing for analyzing multiple documents at once.
*   [ ] More detailed similarity analysis (e.g., highlighting common/different sections).
*   [ ] Option to choose different pre-trained LLMs from a dropdown.
*   [ ] User authentication and a personal document library.
*   [ ] Dockerization for easier deployment and environment consistency.
*   [ ] More sophisticated text preprocessing options configurable by the user.
*   [ ] Integration with cloud storage for document uploads.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/your-repo-name/issues) *(replace with your actual issues page link)*. If you'd like to contribute:

1.  Fork the Project.
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the Branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

Please ensure your code adheres to good practices and include comments where necessary.

## License

Distributed under the MIT License. See `LICENSE.md` for more information.
*(You will need to create a `LICENSE.md` file in your repository. A common choice is the MIT License. You can find templates easily online, e.g., [choosealicense.com](https://choosealicense.com/licenses/mit/))*
