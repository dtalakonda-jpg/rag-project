# RAG Intellect AI

A lightweight, purely Python-based Retrieval-Augmented Generation (RAG) dashboard built with Streamlit and the Google Gemini API. This project is designed to run with zero heavy dependencies (no LangChain, Torch, or FAISS), avoiding common local deployment issues like `WinError 1114` or `ModuleNotFound` while facilitating easy setup and deployment.

## Features

- **Document Analysis**: Upload PDF, TXT, and DOCX files for knowledge extraction and processing.
- **Pure Python Vector Search**: Implements mathematical cosine similarity search, avoiding heavy C++ vector database bindings.
- **Gemini Powered**: Leverages the `gemini-2.5-flash` model for intelligent text generation and multi fallback embedding models (`text-embedding-004`).
- **Modern UI**: A responsive, glassmorphic aesthetics interface that offers a premium user experience.
- **Secure Configuration**: Environment variable management ready for production deployment.

## Installation

1. **Clone the repository.**
2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up Environment Variables:**
   Create a `.env` file in the project root and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
   *(You can copy `.env.example` and rename it to `.env`)*

## Running Locally

Execute the following command to start the Streamlit server:

```bash
streamlit run app.py
```

## Deployment Support

This project is configured for easy deployment on platforms like **Render**, **Streamlit Community Cloud**, or **Heroku**.

**On Render (Web Service):**
1. Connect your GitHub repository to Render and create a new **Web Service**.
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `streamlit run app.py --server.port $PORT`
4. **Environment Variables**: Add your `GEMINI_API_KEY` in the Render dashboard.
