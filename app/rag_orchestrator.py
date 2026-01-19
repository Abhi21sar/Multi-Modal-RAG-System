"""
Multimodal RAG System - Main Orchestrator
==========================================

End-to-end pipeline integrating:
1. CLIP embeddings (unified text/image space)
2. Qdrant vector store (HNSW indexing)
3. LangGraph retrieval workflow
4. Multimodal LLM generation (GPT-4o)

This is the main entry point for the RAG system.

Usage:
    from app.rag_orchestrator import MultimodalRAG
    
    rag = MultimodalRAG()
    result = rag.query("What is mentioned in the document?")
    print(result["answer"])

Author: Abhishek Gurjar
"""

import os
import logging
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from app.embeddings.clip_embedder import CLIPEmbedder
from app.vector_stores.qdrant_store import QdrantVectorStore, Document
from app.workflows.retrieval_graph import retrieve
from app.generation.multimodal_llm import MultimodalGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultimodalRAG:
    """
    Complete multimodal RAG system orchestrator.
    
    This class provides a simple interface for:
    - Indexing documents (text, images, PDFs)
    - Querying the knowledge base
    - Retrieving relevant context
    - Generating grounded answers
    """
    
    def __init__(
        self,
        collection_name: str = "multimodal_rag",
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        model_name: str = "gpt-4o",
        enable_reranking: bool = False,
        enable_reflection: bool = False
    ):
        """
        Initialize the multimodal RAG system.
        
        Args:
            collection_name: Qdrant collection name
            qdrant_url: Qdrant server URL (defaults to :memory: for testing)
            qdrant_api_key: Qdrant Cloud API key (optional)
            model_name: LLM model for generation
            enable_reranking: Use Cohere reranker (requires COHERE_API_KEY)
            enable_reflection: Enable self-reflection quality checks
        """
        logger.info("Initializing Multimodal RAG System...")
        
        # Configuration
        self.collection_name = collection_name
        self.model_name = model_name
        
        # 1. Initialize CLIP Embedder
        logger.info("Loading CLIP embedder...")
        self.embedder = CLIPEmbedder()
        
        # 2. Initialize Qdrant Vector Store
        logger.info("Connecting to Qdrant...")
        if qdrant_url is None:
            qdrant_url = os.getenv("QDRANT_URL", ":memory:")
        if qdrant_api_key is None:
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        self.vector_store = QdrantVectorStore(
            collection_name=collection_name,
            embedding_dim=self.embedder.embedding_dim,
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        # 3. Initialize Reranker (optional)
        self.reranker = None
        if enable_reranking:
            try:
                import cohere
                cohere_key = os.getenv("COHERE_API_KEY")
                if cohere_key:
                    self.reranker = cohere.Client(api_key=cohere_key)
                    logger.info("Cohere reranker enabled")
                else:
                    logger.warning("COHERE_API_KEY not found, reranking disabled")
            except ImportError:
                logger.warning("Cohere library not installed, reranking disabled")
        
        # 4. Initialize Generator
        logger.info(f"Loading {model_name} generator...")
        self.generator = MultimodalGenerator(
            model_name=model_name,
            enable_reflection=enable_reflection
        )
        
        logger.info("✅ Multimodal RAG System initialized successfully!")
        logger.info(f"   - Vector Store: {qdrant_url}")
        logger.info(f"   - Collection: {collection_name}")
        logger.info(f"   - LLM: {model_name}")
        logger.info(f"   - Reranking: {enable_reranking}")
        logger.info(f"   - Reflection: {enable_reflection}")
    
    def index_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> List[str]:
        """
        Index documents into the vector store.
        
        Args:
            documents: List of Document objects (must have embeddings)
            batch_size: Batch size for insertion
            show_progress: Show progress bar
        
        Returns:
            List of indexed document IDs
        """
        logger.info(f"Indexing {len(documents)} documents...")
        
        # Ensure all documents have embeddings
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id} has no embedding. Call embed_documents() first.")
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(
            documents=documents,
            batch_size=batch_size
        )
        
        logger.info(f"✅ Indexed {len(doc_ids)} documents")
        return doc_ids
    
    def embed_documents(
        self,
        documents: List[Document],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Document]:
        """
        Generate embeddings for documents.
        
        Args:
            documents: List of Document objects (content only)
            batch_size: Batch size for embedding
            show_progress: Show progress bar
        
        Returns:
            Documents with embeddings populated
        """
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        
        # Separate by modality
        text_docs = [doc for doc in documents if doc.modality in ["text", "hybrid"]]
        image_docs = [doc for doc in documents if doc.modality == "image"]
        
        # Embed text documents
        if text_docs:
            texts = [doc.content for doc in text_docs]
            embeddings = self.embedder.batch_embed_texts(
                texts=texts,
                batch_size=batch_size,
                show_progress=show_progress
            )
            for doc, emb in zip(text_docs, embeddings):
                doc.embedding = emb
            logger.info(f"   ✓ Embedded {len(text_docs)} text documents")
        
        # Embed image documents (if any)
        if image_docs:
            image_paths = [doc.source for doc in image_docs]  # Assumes source is path
            embeddings = self.embedder.batch_embed_images(
                image_paths=image_paths,
                batch_size=16,  # Smaller batch for images
                show_progress=show_progress
            )
            for doc, emb in zip(image_docs, embeddings):
                doc.embedding = emb
            logger.info(f"   ✓ Embedded {len(image_docs)} image documents")
        
        logger.info("✅ Embedding complete")
        return documents
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        return_sources: bool = True,
        return_context: bool = False
    ) -> Dict:
        """
        Query the RAG system end-to-end.
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
            return_sources: Include source metadata in response
            return_context: Include full context in response
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"Processing query: '{query}'")
        
        # 1. Retrieval (LangGraph workflow)
        retrieval_results = retrieve(
            query=query,
            embedder=self.embedder,
            vector_store=self.vector_store,
            reranker=self.reranker,
            top_k=top_k
        )
        
        logger.info(f"   ✓ Retrieved {retrieval_results['num_results']} documents in {retrieval_results['retrieval_time_ms']:.2f}ms")
        
        # 2. Generation
        generation_results = self.generator.generate(
            query=query,
            context=retrieval_results["context"],
            sources=retrieval_results["sources"]
        )
        
        logger.info(f"   ✓ Generated answer in {generation_results['generation_time_ms']:.2f}ms")
        logger.info(f"   ✓ Confidence: {generation_results['confidence']}")
        
        # 3. Build response
        response = {
            "answer": generation_results["answer"],
            "confidence": generation_results["confidence"],
            "metadata": {
                "retrieval_time_ms": retrieval_results["retrieval_time_ms"],
                "generation_time_ms": generation_results["generation_time_ms"],
                "total_time_ms": retrieval_results["retrieval_time_ms"] + generation_results["generation_time_ms"],
                "num_sources": len(retrieval_results["sources"]),
                "intent": retrieval_results.get("intent", "unknown")
            }
        }
        
        if return_sources:
            response["sources"] = retrieval_results["sources"]
        
        if return_context:
            response["context"] = retrieval_results["context"]
        
        if generation_results.get("quality_score"):
            response["quality_score"] = generation_results["quality_score"]
        
        logger.info(f"✅ Query completed in {response['metadata']['total_time_ms']:.2f}ms")
        
        return response
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        collection_info = self.vector_store.get_collection_info()
        
        return {
            "collection_name": self.collection_name,
            "total_vectors": collection_info["vectors_count"],
            "indexed_vectors": collection_info["indexed_vectors_count"],
            "embedding_dim": self.embedder.embedding_dim,
            "model": self.model_name,
            "status": collection_info["status"]
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_rag_system(
    use_cloud: bool = False,
    enable_reranking: bool = False,
    enable_reflection: bool = False
) -> MultimodalRAG:
    """
    Factory function to create RAG system with sensible defaults.
    
    Args:
        use_cloud: Use Qdrant Cloud (requires QDRANT_URL and QDRANT_API_KEY)
        enable_reranking: Enable Cohere reranker
        enable_reflection: Enable answer quality checks
    
    Returns:
        Initialized MultimodalRAG instance
    """
    qdrant_url = os.getenv("QDRANT_URL")
    
    if use_cloud and not qdrant_url:
        raise ValueError("QDRANT_URL not set for cloud usage")
    
    if not qdrant_url:
        qdrant_url = ":memory:"  # Fallback to defaults
    
    return MultimodalRAG(
        qdrant_url=qdrant_url,
        enable_reranking=enable_reranking,
        enable_reflection=enable_reflection
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Initialize RAG system (in-memory for testing)
    rag = create_rag_system()
    
    # Create sample documents
    sample_docs = [
        Document(
            content="The quick brown fox jumps over the lazy dog. This is a famous pangram.",
            modality="text",
            source="sample1.txt",
            metadata={"category": "linguistics"}
        ),
        Document(
            content="Python is a high-level programming language known for its simplicity.",
            modality="text",
            source="sample2.txt",
            metadata={"category": "programming"}
        ),
    ]
    
    # Embed and index
    embedded_docs = rag.embed_documents(sample_docs)
    rag.index_documents(embedded_docs)
    
    # Query
    result = rag.query("What is a pangram?", top_k=3)
    
    print("\n" + "="*80)
    print("QUERY RESULT")
    print("="*80)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nConfidence: {result['confidence']}")
    print(f"\nProcessing Time: {result['metadata']['total_time_ms']:.2f}ms")
    print(f"  - Retrieval: {result['metadata']['retrieval_time_ms']:.2f}ms")
    print(f"  - Generation: {result['metadata']['generation_time_ms']:.2f}ms")
    print(f"\nSources: {result['metadata']['num_sources']}")
    
    # Show statistics
    stats = rag.get_stats()
    print(f"\nSystem Stats:")
    print(f"  - Total Vectors: {stats['total_vectors']}")
    print(f"  - Collection: {stats['collection_name']}")
    print(f"  - Model: {stats['model']}")
