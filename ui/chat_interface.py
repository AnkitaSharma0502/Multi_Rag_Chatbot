"""
Chat Interface Module
=====================
 Main chat interface logic.

"""

import streamlit as st
from typing import Generator, Optional, List

from core.document_processor import DocumentProcessor
from core.vector_store import VectorStoreManager
from core.chain import RAGChain
from tools.tavily_search import TavilySearchTool, HybridSearchManager
from ui.components import save_uploaded_file


class ChatInterface:
    """
    Main chat interface orchestrator.

    Coordinates between:
    - Document processing
    - Vector store
    - RAG chain
    - Web search
    """

    def __init__(self):
        """Initialize chat interface components."""
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStoreManager()
        self.rag_chain: Optional[RAGChain] = None
        self.tavily_search = TavilySearchTool()
        self.hybrid_search: Optional[HybridSearchManager] = None

        # Fix 2: Store last sources internally instead of re-querying
        self._last_sources: List[str] = []

    def process_uploaded_files(self, uploaded_files) -> int:
        """
        Process uploaded files and add to vector store.

        Args:
            uploaded_files: List of Streamlit UploadedFile objects

        Returns:
            Number of chunks processed
        """
        all_chunks = []

        for uploaded_file in uploaded_files:
            # Save file temporarily
            file_path = save_uploaded_file(uploaded_file)

            # Process the document
            chunks = self.doc_processor.process(file_path)

            # Add source metadata
            for chunk in chunks:
                chunk.metadata["source"] = uploaded_file.name

            all_chunks.extend(chunks)

            # Track uploaded files in sidebar
            if uploaded_file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(uploaded_file.name)

        # Add to vector store
        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            st.session_state.vector_store_initialized = True

        return len(all_chunks)

    def initialize_rag_chain(self):
        """Initialize the RAG chain after documents are loaded."""
        if self.vector_store.is_initialized:
            self.rag_chain = RAGChain(self.vector_store)
            self.hybrid_search = HybridSearchManager(
                self.vector_store,
                self.tavily_search
            )

    def _build_context(self, query: str, use_web_search: bool) -> str:
        """
        Build context string from documents and/or web search.
        Args:
            query: User's question
            use_web_search: Whether to include web search

        Returns:
            Formatted context string and updates _last_sources
        """
        context_parts = []
        self._last_sources = []

        # Document results
        if self.vector_store.is_initialized:
            doc_results = self.vector_store.search(query)

            if doc_results:
                context_parts.append("=== From Your Documents ===")
                for doc in doc_results:
                    # Fix 3: Correct citation format
                    title = doc.metadata.get("title", "Unknown")
                    chunk_index = doc.metadata.get("chunk_index", "?")
                    citation = f"[Doc] {title} – Chunk{chunk_index}"
                    context_parts.append(f"{citation}\n{doc.page_content}")

                    # Store sources for get_last_sources()
                    self._last_sources.append(citation)

        # Web search results
        if use_web_search:
            web_results = self.tavily_search.search(query)
            if web_results:
                context_parts.append("\n=== From Web Search ===")
                context_parts.append(f"[Web] Tavily: {web_results}")
                self._last_sources.append("[Web] Tavily Search")

        return "\n\n".join(context_parts) if context_parts else "No context available."

    def get_response(
        self,
        query: str,
        use_web_search: bool = False
    ) -> Generator[str, None, None]:
        """
        Get a streaming response for a query.

        Args:
            query: User's question
            use_web_search: Whether to include web search

        Yields:
            Response chunks
        """
        # Initialize RAG chain if needed
        if self.rag_chain is None and self.vector_store.is_initialized:
            self.initialize_rag_chain()

        # No documents and no web search
        if not self.vector_store.is_initialized and not use_web_search:
            yield "Please upload some documents first, or enable web search to get started!"
            return

        # Fix 1: Build context once and reuse chain.py's LLM
        context = self._build_context(query, use_web_search)

        # Use existing RAG chain instead of creating new LLM
        if self.rag_chain:
            for chunk in self.rag_chain.generate_stream(query, context):
                yield chunk
        else:
            # Web search only — no documents uploaded
            # Initialize a temporary chain just for LLM access
            self.rag_chain = RAGChain(self.vector_store)
            for chunk in self.rag_chain.generate_stream(query, context):
                yield chunk

    def get_last_sources(self) -> List[str]:
        """
        Get sources from the last query.

        Returns:
            List of source citations from last get_response() call
        """
        return self._last_sources