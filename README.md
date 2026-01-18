# 🤖 NLP - ChillBot

ChillBot is a specialized Streamlit-based AI assistant designed to help you interact with your PDF documents. It goes beyond simple text extraction by handling scanned images (OCR), extracting tables with automatic visualization, and providing advanced NLP features like summarization and translation.

## Key Features

- **Multi-PDF Interaction**: Upload up to 3 PDFs (max 5MB each) and chat with them simultaneously.
- **Intelligent RAG**: Uses LangChain and FAISS vector stores with Hugging Face embeddings for accurate document retrieval.
- **OCR Support**: Automatically detects non-readable PDF pages and uses Tesseract OCR to extract text.
- **Table Extraction & Visualization**: Identifies tables within your PDFs and automatically generates bar charts for numeric data.
- **Summarization**: Enter `[Summary]` to get a concise summary of the entire document using the BART model.
- **Translation**: Enter `[Translate]` to translate the document content into Indonesian.
- **Chat History**: Save and download your conversation history for future reference.

## Prerequisites: OCR & PDF Processing

To enable OCR (for scanned PDFs) and PDF-to-image conversion, you must install the following external tools on your Windows machine:

### 1. Tesseract OCR
Tesseract is used to read text from images.
- **Download**: [UB-Mannheim Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki).
- **Install**: Run the installer and install it to the default path: `C:\Program Files\Tesseract-OCR`.
- **Code Check**: In `app.py`, ensure the following line matches your actual installation path:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```

### 2. Poppler
Poppler is required to convert PDF pages into images for the OCR process.
- **Download**: [Latest Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases).
- **Setup**: Extract the downloaded `.zip` file to a permanent location (e.g., `C:\poppler`).
- **Code Check**: In `app.py`, update the `poppler_path` inside the `get_pdf_text_and_tables` function to point to your extracted `bin` folder:
  ```python
  poppler_path = r"C:\path\to\your\poppler\Library\bin"
  ```

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/NLP---ChillBot.git
   cd NLP---ChillBot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API Key**:
   Open `app.py` and replace `"YOUR API KEY"` with your actual Hugging Face API Token:
   ```python
   os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_your_token_here"
   ```

4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## How to Use
1. Select your preferred LLM model in the sidebar.
2. Upload your PDF(s) and click **Process**.
3. View any extracted tables and visualizations in the main area.
4. Use the chat input to ask questions or type commands like `Summary` or `Translate`.
