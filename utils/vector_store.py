from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from PyPDF2 import PdfReader

class VectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        """
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def create_vector_store(self, documents: List[str], persist_directory: str = "data/faiss_db"):
        """
        Create a vector store from documents using FAISS
        """
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        # Split documents into chunks
        splits = text_splitter.create_documents(documents)
        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        # Optionally persist to disk
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.vector_store.save_local(persist_directory)
        return self.vector_store

    def similarity_search(self, query: str, k: int = 4) -> List[Dict]:
        """
        Perform similarity search
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
        return self.vector_store.similarity_search(query, k=k)
