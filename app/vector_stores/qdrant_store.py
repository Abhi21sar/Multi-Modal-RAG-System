"""
Qdrant Vector Store for Multimodal RAG
=======================================

Production-grade vector store implementation using Qdrant with:
- HNSW indexing for sub-second retrieval
- Scalar quantization for memory efficiency
- Distributed architecture support
- Rich metadata filtering
- Batch operations for high throughput

Performance Targets:
- Latency: <100ms for top-10 retrieval (10M vectors)
- Throughput: >500 QPS
- Memory: ~4GB per 1M 768-dim vectors (with quantization)

Author: Abhishek Gurjar
"""

import uuid
import numpy as np
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    HnswConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    OptimizersConfigDiff,
    SearchParams
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Structured document with embedding and metadata."""
    content: str
    modality: str = "text"  # "text", "image", "hybrid"
    source: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class QdrantVectorStore:
    """
    Production-grade vector store using Qdrant.
    
    Features:
    - HNSW indexing for fast approximate nearest neighbor search
    - Scalar quantization for 75% memory reduction
    - Rich metadata filtering
    - Batch operations
    - Distributed architecture support
    """
    
    def __init__(
        self,
        collection_name: str = "multimodal_rag",
        embedding_dim: int = 768,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        **kwargs
    ):
        """
        Initialize Qdrant vector store.
        
        Args:
            collection_name: Name of the collection
            embedding_dim: Dimension of embeddings (768 for CLIP)
            url: Qdrant server URL (use ":memory:" for local testing)
            api_key: API key for Qdrant Cloud
            prefer_grpc: Use gRPC for better performance (if available)
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Initialize client
        # Initialize client
        if url == ":memory:":
            logger.info("Using in-memory Qdrant storage (for testing only)")
            self.client = QdrantClient(":memory:")
        elif url.startswith(":") and not url.startswith("http"):
            # Handle local path syntax like ":./qdrant_data:"
            path = url.strip(":")
            logger.info(f"Using local persistent Qdrant storage at: {path}")
            self.client = QdrantClient(path=path)
        else:
            logger.info(f"Connecting to Qdrant at {url}")
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
                **kwargs
            )
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create collection with optimized configuration if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,  # For L2-normalized vectors
                    hnsw_config=HnswConfigDiff(
                        m=16,              # Connections per layer (balance: speed vs accuracy)
                        ef_construct=200,  # Indexing quality (higher = better but slower)
                    ),
                    on_disk=True,  # Enable disk storage for large collections
                ),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=10000,  # Index after 10k vectors
                ),
            )
            
            # Enable quantization for memory efficiency
            self._enable_quantization()
            
            logger.info(f"Collection '{self.collection_name}' created successfully")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")
    
    def _enable_quantization(self):
        """Enable scalar quantization to reduce memory usage by ~75%."""
        try:
            self.client.update_collection(
                collection_name=self.collection_name,
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,  # Ignore outliers
                        always_ram=True  # Keep quantized vectors in RAM
                    )
                )
            )
            logger.info("Scalar quantization enabled (INT8, 75% memory reduction)")
        except Exception as e:
            logger.warning(f"Could not enable quantization: {e}")
    
    def add_documents(
        self, 
        documents: List[Document],
        batch_size: int = 100,
        parallel: int = 1
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects
            batch_size: Number of documents to insert per batch
            parallel: Number of parallel workers (for large batches)
        
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []
        
        logger.info(f"Adding {len(documents)} documents to collection")
        
        # Convert documents to Qdrant points
        points = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id} has no embedding")
            
            point = PointStruct(
                id=doc.id,
                vector=doc.embedding.tolist(),
                payload={
                    "content": doc.content,
                    "source": doc.source,
                    "modality": doc.modality,
                    "metadata": doc.metadata
                }
            )
            points.append(point)
        
        # Batch insert for efficiency
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True  # Wait for indexing to complete
            )
            logger.info(f"Inserted batch {i // batch_size + 1}/{(len(points) - 1) // batch_size + 1}")
        
        logger.info(f"Successfully added {len(documents)} documents")
        return [doc.id for doc in documents]
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None,
        score_threshold: Optional[float] = None,
        search_params: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Semantic search using vector similarity.
        
        Args:
            query_vector: Query embedding (768-dim for CLIP)
            top_k: Number of results to return
            filter_dict: Metadata filters (e.g., {"modality": "text"})
            score_threshold: Minimum similarity score (0-1 for cosine)
            search_params: Advanced search parameters (e.g., {"hnsw_ef": 128})
        
        Returns:
            List of search results with content, metadata, and scores
        """
        # Build filter if provided
        query_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}" if key not in ["modality", "source", "content"] else key,
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)
        
        # Prepare search params
        sp = SearchParams(hnsw_ef=128)  # Balance: speed vs recall
        if search_params:
            sp = SearchParams(**search_params)
        
        # Execute search (using query_points for Qdrant v1.7+)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            search_params=sp
        ).points
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "content": result.payload.get("content", ""),
                "source": result.payload.get("source", ""),
                "modality": result.payload.get("modality", "unknown"),
                "metadata": result.payload.get("metadata", {})
            })
        
        return formatted_results
    
    def hybrid_search(
        self,
        query_vector: np.ndarray,
        filters: List[Dict],
        top_k: int = 5,
        rrf_k: int = 60  # Reciprocal Rank Fusion parameter
    ) -> List[Dict]:
        """
        Advanced hybrid search combining multiple filter strategies with RRF.
        
        Args:
            query_vector: Query embedding
            filters: List of filter dictionaries for different strategies
            top_k: Final number of results
            rrf_k: RRF constant (default: 60)
        
        Returns:
            Reranked results using Reciprocal Rank Fusion
        """
        all_results = []
        
        # Execute multiple searches with different filters
        for filter_dict in filters:
            results = self.search(
                query_vector=query_vector,
                top_k=top_k * 2,  # Retrieve more for better fusion
                filter_dict=filter_dict
            )
            all_results.append(results)
        
        # Reciprocal Rank Fusion
        doc_scores = {}
        for results in all_results:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc["id"]
                rrf_score = 1.0 / (rrf_k + rank)
                
                if doc_id in doc_scores:
                    doc_scores[doc_id]["score"] += rrf_score
                else:
                    doc_scores[doc_id] = {
                        **doc,
                        "score": rrf_score
                    }
        
        # Sort by fused score
        fused_results = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]
        
        return fused_results
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=document_ids
        )
        logger.info(f"Deleted {len(document_ids)} documents")
        return True
    
    def get_collection_info(self) -> Dict:
        """Get collection statistics."""
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status,
            "optimizer_status": info.optimizer_status
        }
    
    def optimize(self):
        """Manually trigger collection optimization."""
        logger.info("Starting collection optimization...")
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=0  # Force immediate indexing
            )
        )
        logger.info("Optimization completed")


# Example usage
if __name__ == "__main__":
    # Initialize vector store (in-memory for testing)
    vector_store = QdrantVectorStore(url=":memory:")
    
    # Create sample documents
    documents = [
        Document(
            content="The quick brown fox jumps over the lazy dog",
            embedding=np.random.randn(768).astype(np.float32),
            modality="text",
            source="sample.txt",
            metadata={"page": 1, "section": "intro"}
        ),
        Document(
            content="A cat sitting on a mat",
            embedding=np.random.randn(768).astype(np.float32),
            modality="text",
            source="sample.txt",
            metadata={"page": 2, "section": "body"}
        ),
    ]
    
    # Add documents
    doc_ids = vector_store.add_documents(documents)
    print(f"Added documents: {doc_ids}")
    
    # Search
    query_vector = np.random.randn(768).astype(np.float32)
    results = vector_store.search(query_vector, top_k=2)
    print(f"\nSearch results:")
    for result in results:
        print(f"  - {result['content'][:50]}... (score: {result['score']:.4f})")
    
    # Get collection info
    info = vector_store.get_collection_info()
    print(f"\nCollection info: {info}")
