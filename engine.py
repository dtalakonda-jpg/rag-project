import os
import google.generativeai as genai
from pypdf import PdfReader
import numpy as np
from dotenv import load_dotenv
import docx2txt

load_dotenv()

class RAGEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.chunks = []
        self.embeddings = []

    def _get_embedding(self, content, task_type="retrieval_document"):
        """Try multiple embedding models as fallbacks and report the exact error."""
        errors = []
        model_names = [
            "models/text-embedding-004", "text-embedding-004",
            "models/embedding-001", "embedding-001",
            "models/gemini-embedding-001", "gemini-embedding-001"
        ]
        
        for model_name in model_names:
            try:
                response = genai.embed_content(
                    model=model_name,
                    content=content,
                    task_type=task_type
                )
                return response['embedding']
            except Exception as e:
                errors.append(f"{model_name}: {str(e)}")
                continue
        
        error_msg = "\n".join(errors)
        raise Exception(f"Failed to find a working model. Details:\n{error_msg}")

    def process_documents(self, uploaded_files):
        """
        Extract text, chunk it, and get embeddings using ONLY Gemini API.
        Zero dependencies on LangChain, Torch, or FAISS.
        """
        all_text = ""
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith(".pdf"):
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    all_text += page.extract_text() or ""
            elif uploaded_file.name.endswith(".txt"):
                all_text += uploaded_file.read().decode("utf-8")
            elif uploaded_file.name.endswith(".docx") or uploaded_file.name.endswith(".doc"):
                all_text += docx2txt.process(uploaded_file)

        if not all_text:
            return False

        # Simple Chunking (Pure Python)
        chunk_size = 1000
        overlap = 200
        self.chunks = []
        for i in range(0, len(all_text), chunk_size - overlap):
            self.chunks.append(all_text[i:i + chunk_size])

        # Get Embeddings using Resilient Model Selector
        if self.chunks:
            self.embeddings = self._get_embedding(self.chunks, task_type="retrieval_document")
        return True

    def query(self, question: str):
        """
        Pure Python Vector Search (Cosine Similarity) and Gemini Generation.
        """
        if not self.chunks:
            return "Please upload documents first.", []

        # 1. Embed the query with fallback
        query_emb = self._get_embedding(question, task_type="retrieval_query")

        # 2. Simple Similarity Search (Dot Product on normalized vectors)
        similarities = [np.dot(query_emb, doc_emb) for doc_emb in self.embeddings]
        top_indices = np.argsort(similarities)[-3:][::-1] # Get top 3
        
        context = "\n\n".join([self.chunks[i] for i in top_indices])
        
        # 3. Generate Answer
        prompt = f"""
        Answer the following question using ONLY the provided context. 
        If you don't know the answer, say "I don't know".
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        response = self.model.generate_content(prompt)
        
        # Return answer + mock source objects for the UI
        class Source:
            def __init__(self, content): self.page_content = content
        
        sources = [Source(self.chunks[i]) for i in top_indices]
        
        return response.text, sources
