"""
Tavily Search Tool Module
=========================
This module provides web search capabilities via Tavily.

"""

import os
from typing import List, Optional, Literal
from langchain_tavily import TavilySearch

from config.settings import settings


class TavilySearchTool:
    """
    Web search tool using Tavily API.
    
    Use this when:
    - User asks about current events
    - Document search doesn't have relevant information
    - User explicitly asks to search the web
    """

    def __init__(
        self,
        max_results: int = 3,
        topic: Literal["general", "news", "finance"] = "general"
    ):
        self.max_results = max_results
        self.topic = topic

        # Set Tavily API key in environment (required by langchain-tavily)
        os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY

        # Initialize Tavily search
        self._search = TavilySearch(
            max_results=self.max_results,
            topic=self.topic
        )

    @property
    def tool(self) -> TavilySearch:
        """Get the Tavily search tool instance."""
        return self._search

    def search(self, query: str) -> str:
        """Perform a web search and return formatted string."""
        results = self._search.invoke(query)
        return self._format_results(results)

    def _format_results(self, results: dict) -> str:
        """Format Tavily results dictionary into readable string."""
        if not results:
            return "No search results found."

        formatted_parts = []

        # Add answer if available
        if results.get("answer"):
            formatted_parts.append(f"Summary: {results['answer']}")

        # Add individual results
        if results.get("results"):
            for i, result in enumerate(results["results"], 1):
                title = result.get("title", "No title")
                content = result.get("content", "No content")
                url = result.get("url", "")
                formatted_parts.append(
                    f"[Web] Tavily: \"{title}\"\n{content}\nSource: {url}"
                )

        return "\n\n".join(formatted_parts) if formatted_parts else "No results found."

    def search_with_context(self, query: str) -> dict:
        """Perform a web search and return structured results."""
        raw_results = self._search.invoke(query)

        return {
            "query": query,
            "results": raw_results,
            "formatted": self._format_results(raw_results),
            "source": "tavily_web_search"
        }


class HybridSearchManager:
    """
    Manages hybrid search: combines document search with web search.

    Strategy:
    1. First, search in local documents
    2. If results are insufficient, augment with web search
    """

    def __init__(
        self,
        vector_store_manager,
        tavily_tool: TavilySearchTool = None
    ):
        self.vector_store = vector_store_manager
        self.tavily = tavily_tool or TavilySearchTool()

    def search(
        self,
        query: str,
        use_web_search: bool = False,
        doc_k: int = 3
    ) -> dict:
        """Perform hybrid search combining documents and web."""
        results = {
            "query": query,
            "document_results": [],
            "web_results": None
        }

        # Document search (if vector store is initialized)
        if self.vector_store.is_initialized:
            docs = self.vector_store.search(query, k=doc_k)
            results["document_results"] = docs

        # Web search (if enabled)
        if use_web_search:
            web_results = self.tavily.search(query)
            results["web_results"] = web_results

        return results

    def format_hybrid_context(
        self,
        doc_results: List,
        web_results: Optional[str] = None
    ) -> str:
        """Format hybrid search results into context string with citations."""
        context_parts = []

        # Add document context
        if doc_results:
            context_parts.append("=== From Your Documents ===")
            for i, doc in enumerate(doc_results, 1):
                title = doc.metadata.get("title", "Unknown")
                chunk_index = doc.metadata.get("chunk_index", "?")

                # Citation format: [Doc] myfile – Chunk3
                citation = f"[Doc] {title} – Chunk{chunk_index}"
                context_parts.append(f"{citation}\n{doc.page_content}")

        # Add web context
        if web_results:
            context_parts.append("\n=== From Web Search ===")
            # Citation format: [Web] Tavily: "Page Title"
            context_parts.append(f"[Web] Tavily: {web_results}")

        return "\n\n".join(context_parts) if context_parts else "No context available."