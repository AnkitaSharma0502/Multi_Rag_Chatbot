
"""
Document Processor Module
=========================
Step 1: This module handles loading and splitting documents.

SOLID Principle: Single Responsibility Principle (SRP)
- Class Role: process documents (load + split)
"""

from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader #langchain tools to read pdf,txt
from langchain_text_splitters import RecursiveCharacterTextSplitter #cut long text to smalelr chunks

from config.settings import settings   


class DocumentProcessor:
    """
    Handles document loading and text splitting.
    Supports: Text files (.txt) and PDF files (.pdf)
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None): #expicitly handeling chunk sizes if not then use default from settings.py
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,                           #small overlap between chunks so context isn't lost at the edges. For example if overlap is 50, the end of chunk 1 repeats at the start of chunk 2.
            length_function=len,
            separators=["\n\n", "\n", " ", ""]                           #"\n\n"  → first try splitting at blank lines (best)
            #                                                             "\n"    → then at single line breaks
            #                                                             " "     → then at spaces
            #                                                             ""      → last resort, cut anywhere
        )

    def load_document(self, file_path: str) -> List[Document]:
        """Load a document from file path (.txt or .pdf)."""
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            source_type = "txt"
        elif extension == ".pdf":
            loader = PyPDFLoader(file_path)
            source_type = "pdf"
        else:
            raise ValueError(f"Unsupported file type: {extension}. Use .txt or .pdf")

        documents = loader.load()

        # Tag each document with source_type and title
        for doc in documents:
            doc.metadata["source_type"] = source_type
            doc.metadata["title"] = path.stem  # filename without extension

        return documents

    def load_from_text(self, text: str, metadata: dict = None) -> List[Document]:
        """Create a document from raw text (used for Wikipedia content)."""
        metadata = metadata or {}              # if no metadata was passed, start with an empty dictionary instead of None
        metadata.setdefault("source_type", "wikipedia")
        return [Document(page_content=text, metadata=metadata)]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks and add chunk_index to each chunk."""
        chunks = self.text_splitter.split_documents(documents)

        # Add chunk_index so we know which chunk came from where
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        return chunks

    def process(self, file_path: str) -> List[Document]:
        """Full pipeline: load a file and split it into chunks."""
        documents = self.load_document(file_path)
        chunks = self.split_documents(documents)
        return chunks

    def process_text(self, text: str, metadata: dict = None) -> List[Document]:
        """Full pipeline: take raw text and split it into chunks."""
        documents = self.load_from_text(text, metadata)
        chunks = self.split_documents(documents)
        return chunks