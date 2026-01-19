"""
LangGraph Retrieval Workflow
==============================

Stateful retrieval pipeline using LangGraph for orchestrating:
1. Query analysis and intent detection
2. Multi-strategy retrieval (semantic + filtered)
3. Reranking for precision
4. Context fusion and preparation

This provides a robust, observable, and extensible retrieval system.

Author: Abhishek Gurjar
"""

from typing import TypedDict, List, Dict, Any, Annotated
import operator
import logging
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# State Definitions
# ============================================================================

class RetrievalState(TypedDict):
    """State passed through the retrieval workflow."""
    # Input
    query: str
    top_k: int
    
    # Intermediate
    query_vector: np.ndarray
    intent: str  # "factual", "comparison", "summary", etc.
    filters: Dict[str, Any]
    modality_preference: str  # "text", "image", "both"
    
    # Retrieval results
    dense_results: List[Dict]
    filtered_results: List[Dict]
    
    # Final output
    reranked_results: List[Dict]
    context: str
    sources: List[Dict]
    
    # Metadata
    error: str
    retrieval_time_ms: float


# ============================================================================
# Workflow Nodes
# ============================================================================

def analyze_query(state: RetrievalState) -> RetrievalState:
    """
    Analyze user query to extract intent and preferences.
    
    This node determines:
    - Query intent (factual, comparative, etc.)
    - Modality preference (text, images, both)
    - Any implicit filters (dates, types, etc.)
    """
    query = state["query"].lower()
    
    # Simple intent detection (can be upgraded to LLM-based)
    if any(word in query for word in ["compare", "difference", "versus", "vs"]):
        intent = "comparison"
    elif any(word in query for word in ["summary", "summarize", "overview"]):
        intent = "summary"
    elif any(word in query for word in ["how", "why", "what", "when", "where"]):
        intent = "factual"
    else:
        intent = "general"
    
    # Modality preference detection
    if any(word in query for word in ["image", "picture", "photo", "diagram"]):
        modality = "image"
    elif any(word in query for word in ["text", "document", "article"]):
        modality = "text"
    else:
        modality = "both"
    
    logger.info(f"Query analysis: intent={intent}, modality={modality}")
    
    return {
        **state,
        "intent": intent,
        "modality_preference": modality,
        "filters": {}
    }


def dense_retrieval(
    state: RetrievalState,
    embedder: Any,  # CLIPEmbedder instance
    vector_store: Any  # QdrantVectorStore instance
) -> RetrievalState:
    """
    Perform dense semantic retrieval using CLIP embeddings.
    
    This is the primary retrieval strategy using pure vector similarity.
    """
    import time
    start_time = time.time()
    
    query = state["query"]
    top_k = state.get("top_k", 20)
    
    # Generate query embedding
    query_vector = embedder.embed_text(query)
    
    # Search vector store
    results = vector_store.search(
        query_vector=query_vector,
        top_k=top_k,
        score_threshold=0.5  # Filter low-quality matches
    )
    
    retrieval_time = (time.time() - start_time) * 1000
    logger.info(f"Dense retrieval: {len(results)} results in {retrieval_time:.2f}ms")
    
    return {
        **state,
        "query_vector": query_vector,
        "dense_results": results,
        "retrieval_time_ms": retrieval_time
    }


def filtered_retrieval(
    state: RetrievalState,
    vector_store: Any
) -> RetrievalState:
    """
    Apply metadata filters to narrow down results.
    
    This combines semantic search with structured filtering.
    """
    query_vector = state["query_vector"]
    modality_pref = state.get("modality_preference", "both")
    top_k = state.get("top_k", 20)
    
    # Build filter based on modality preference
    filter_dict = {}
    if modality_pref != "both":
        filter_dict["modality"] = modality_pref
    
    # Add any additional filters from state
    filter_dict.update(state.get("filters", {}))
    
    # Search with filters
    if filter_dict:
        results = vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            filter_dict=filter_dict
        )
        logger.info(f"Filtered retrieval: {len(results)} results with filters {filter_dict}")
    else:
        results = state["dense_results"]  # Skip if no filters
        logger.info("Filtered retrieval: No filters applied, using dense results")
    
    return {
        **state,
        "filtered_results": results
    }


def rerank_results(
    state: RetrievalState,
    reranker: Any = None  # Optional: Cohere reranker or cross-encoder
) -> RetrievalState:
    """
    Rerank results for improved precision.
    
    If reranker is provided (Cohere API or cross-encoder), use it.
    Otherwise, use simple score-based ranking.
    """
    query = state["query"]
    candidates = state["filtered_results"]
    top_k = state.get("top_k", 5)
    
    if reranker is None:
        # Simple score-based ranking
        reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]
        logger.info(f"Reranking: Using score-based ranking (no reranker provided)")
    else:
        # Use advanced reranker (Cohere or cross-encoder)
        try:
            # Cohere Rerank API
            documents = [doc["content"] for doc in candidates]
            rerank_results = reranker.rerank(
                query=query,
                documents=documents,
                top_n=top_k
            )
            
            # Map back to original documents
            reranked = []
            for result in rerank_results.results:
                original_doc = candidates[result.index]
                original_doc["rerank_score"] = result.relevance_score
                reranked.append(original_doc)
            
            logger.info(f"Reranking: Used Cohere reranker, top resultado score={reranked[0]['rerank_score']:.4f}")
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, falling back to score-based")
            reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]
    
    return {
        **state,
        "reranked_results": reranked[:top_k]
    }


def fuse_context(state: RetrievalState) -> RetrievalState:
    """
    Prepare final context from reranked results.
    
    This formats the results into a context string for the LLM.
    """
    results = state["reranked_results"]
    
    # Build context with source attribution
    context_parts = []
    sources = []
    
    for i, doc in enumerate(results, start=1):
        source_info = {
            "index": i,
            "source": doc["source"],
            "modality": doc["modality"],
            "score": doc.get("score", 0.0),
            "metadata": doc.get("metadata", {})
        }
        sources.append(source_info)
        
        # Format context entry
        context_entry = f"[Source {i}: {doc['source']}]\n{doc['content']}\n"
        context_parts.append(context_entry)
    
    context = "\n---\n".join(context_parts)
    
    logger.info(f"Context fusion: {len(results)} documents, {len(context)} chars")
    
    return {
        **state,
        "context": context,
        "sources": sources
    }


# ============================================================================
# Workflow Builder
# ============================================================================

def create_retrieval_graph(
    embedder: Any,
    vector_store: Any,
    reranker: Any = None
) -> StateGraph:
    """
    Create the LangGraph retrieval workflow.
    
    Args:
        embedder: CLIPEmbedder instance
        vector_store: QdrantVectorStore instance
        reranker: Optional Cohere reranker
    
    Returns:
        Compiled StateGraph
    """
    # Initialize graph
    workflow = StateGraph(RetrievalState)
    
    # Add nodes with partial application of external dependencies
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node(
        "dense_retrieval",
        lambda state: dense_retrieval(state, embedder, vector_store)
    )
    workflow.add_node(
        "filtered_retrieval",
        lambda state: filtered_retrieval(state, vector_store)
    )
    workflow.add_node(
        "rerank_results",
        lambda state: rerank_results(state, reranker)
    )
    workflow.add_node("fuse_context", fuse_context)
    
    # Define edges (workflow)
    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "dense_retrieval")
    workflow.add_edge("dense_retrieval", "filtered_retrieval")
    workflow.add_edge("filtered_retrieval", "rerank_results")
    workflow.add_edge("rerank_results", "fuse_context")
    workflow.add_edge("fuse_context", END)
    
    # Compile graph
    app = workflow.compile()
    
    logger.info("Retrieval graph compiled successfully")
    return app


# ============================================================================
# Convenience Function
# ============================================================================

def retrieve(
    query: str,
    embedder: Any,
    vector_store: Any,
    reranker: Any = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Convenience function to run the full retrieval pipeline.
    
    Args:
        query: User query string
        embedder: CLIPEmbedder instance
        vector_store: QdrantVectorStore instance
        reranker: Optional reranker
        top_k: Number of final results
    
    Returns:
        Dictionary with context, sources, and metadata
    """
    # Create graph
    graph = create_retrieval_graph(embedder, vector_store, reranker)
    
    # Execute workflow
    initial_state = {
        "query": query,
        "top_k": top_k,
        "error": "",
        "retrieval_time_ms": 0.0
    }
    
    final_state = graph.invoke(initial_state)
    
    return {
        "context": final_state.get("context", ""),
        "sources": final_state.get("sources", []),
        "intent": final_state.get("intent", ""),
        "retrieval_time_ms": final_state.get("retrieval_time_ms", 0.0),
        "num_results": len(final_state.get("reranked_results", []))
    }


# Example usage
if __name__ == "__main__":
    logger.info("LangGraph Retrieval Workflow loaded successfully")
    logger.info("Use create_retrieval_graph() or retrieve() to run workflows")
