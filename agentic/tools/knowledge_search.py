"""Knowledge search tool: RAG-based retrieval over CultPass support articles."""

import sqlite3
import os
import json
import logging
from typing import List, Dict, Any

from langchain.tools import tool

logger = logging.getLogger("uda_hub.tools.knowledge_search")


def _keyword_score(query_tokens: List[str], text: str) -> float:
    """Simple keyword relevance scoring."""
    text_lower = text.lower()
    score = 0.0
    for token in query_tokens:
        occurrences = text_lower.count(token)
        if occurrences > 0:
            score += 1.0 + 0.2 * (occurrences - 1)
    return score


def create_knowledge_search_tool(db_path: str, embeddings=None, vector_store=None):
    """
    Create a knowledge search tool.

    If a vector_store (ChromaDB collection) and embeddings model are provided,
    the tool uses semantic search (RAG). Otherwise it falls back to keyword search.
    """

    @tool
    def knowledge_search(query: str, top_k: int = 3) -> str:
        """
        Search the CultPass knowledge base for relevant support articles.

        Uses semantic search (embeddings + vector DB) when available, otherwise
        falls back to keyword search over the SQLite knowledge table.

        Args:
            query: The search query describing the customer's issue.
            top_k: Maximum number of articles to return (default 3).

        Returns:
            Matching knowledge base articles with IDs, titles, and content.
        """
        # Try vector search first
        if vector_store is not None and embeddings is not None:
            try:
                results = vector_store.similarity_search_with_score(query, k=top_k)
                if results:
                    formatted = f"Found {len(results)} relevant article(s):\n\n"
                    for doc, score in results:
                        article_id = doc.metadata.get("id", "unknown")
                        title = doc.metadata.get("title", "Untitled")
                        category = doc.metadata.get("category", "general")
                        formatted += f"Article {article_id}: {title}\n"
                        formatted += f"Category: {category}\n"
                        formatted += f"Relevance Score: {score:.3f}\n"
                        formatted += f"Content:\n{doc.page_content}\n"
                        formatted += "-" * 50 + "\n"
                    logger.info(
                        "Knowledge search (vector)",
                        extra={
                            "event": "tool_call",
                            "operation": "knowledge_search",
                            "query": query,
                            "outcome": "found",
                            "matches": len(results),
                        },
                    )
                    return formatted
            except Exception:
                pass  # Fall through to keyword search

        # Keyword search fallback
        if not os.path.exists(db_path):
            return f"Error: Database not found at {db_path}. Run 02_core_db_setup.py first."

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, content, category, tags FROM knowledge")
            articles = cursor.fetchall()

            if not articles:
                return "No knowledge base articles found. Run 02_core_db_setup.py to load articles."

            query_tokens = [t.lower() for t in query.split() if len(t) > 2]

            scored: List[Dict[str, Any]] = []
            for article in articles:
                title_score = _keyword_score(query_tokens, article["title"]) * 2.0
                content_score = _keyword_score(query_tokens, article["content"])
                tags_str = article["tags"] or "[]"
                tag_score = _keyword_score(query_tokens, tags_str) * 1.5
                category_score = _keyword_score(query_tokens, article["category"] or "") * 1.5
                total = title_score + content_score + tag_score + category_score

                if total > 0:
                    scored.append({
                        "id": article["id"],
                        "title": article["title"],
                        "content": article["content"],
                        "category": article["category"],
                        "score": total,
                    })

            scored.sort(key=lambda x: x["score"], reverse=True)
            top_results = scored[:top_k]

            if not top_results:
                return (
                    f"No relevant articles found for query: '{query}'. "
                    "The knowledge base may not cover this topic. Consider escalation."
                )

            formatted = f"Found {len(top_results)} relevant article(s):\n\n"
            for r in top_results:
                formatted += f"Article {r['id']}: {r['title']}\n"
                formatted += f"Category: {r['category']}\n"
                formatted += f"Relevance Score: {r['score']:.2f}\n"
                formatted += f"Content:\n{r['content']}\n"
                formatted += "-" * 50 + "\n"

            logger.info(
                "Knowledge search (keyword)",
                extra={
                    "event": "tool_call",
                    "operation": "knowledge_search",
                    "query": query,
                    "outcome": "found",
                    "matches": len(top_results),
                },
            )
            return formatted

        except Exception as e:
            logger.error(
                f"Knowledge search error: {e}",
                extra={"event": "tool_call", "operation": "knowledge_search", "outcome": "error"},
            )
            return f"Error searching knowledge base: {str(e)}"
        finally:
            conn.close()

    return knowledge_search
