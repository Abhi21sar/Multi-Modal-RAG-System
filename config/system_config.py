"""
Configuration File for Multimodal RAG System
=============================================

Centralized configuration for all system components.

Author: Abhishek Gurjar
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmbeddingConfig:
    """CLIP Embedding Configuration"""
    model_name: str = "openai/clip-vit-large-patch14-336"
    embedding_dim: int = 768
    device: Optional[str] = None  # Auto-detect if None
    batch_size_text: int = 32
    batch_size_image: int = 16


@dataclass
class QdrantConfig:
    """Qdrant Vector Store Configuration"""
    # Connection
    url: str = ":memory:"  # Use ":memory:" for testing, "http://localhost:6333" for local Docker
    api_key: Optional[str] = None
    prefer_grpc: bool = False
    
    # Collection
    collection_name: str = "multimodal_rag"
    
    # HNSW Parameters (Performance Tuning)
    hnsw_m: int = 16              # Connections per layer (16-64, higher = better recall)
    hnsw_ef_construct: int = 200  # Indexing quality (100-500, higher = slower index but better quality)
    hnsw_ef_search: int = 128     # Search quality (64-512, higher = slower search but better recall)
    
    # Quantization (Memory Optimization)
    enable_quantization: bool = True  # Reduces memory by 75%
    quantization_type: str = "int8"   # "int8" or "binary"
    
    # Storage
    on_disk: bool = True  # Enable disk storage for large collections
    
    # Performance
    indexing_threshold: int = 10000  # Start indexing after N vectors
    batch_size: int = 100            # Batch size for insertion


@dataclass
class RetrievalConfig:
    """Retrieval Pipeline Configuration"""
    top_k_candidates: int = 20  # Initial retrieval
    top_k_final: int = 5        # After reranking
    score_threshold: float = 0.5  # Minimum similarity score (0-1)
    
    # Hybrid Search
    enable_hybrid_search: bool = False
    rrf_k: int = 60  # Reciprocal Rank Fusion parameter
    
    # Reranking
    enable_reranking: bool = False
    rerank_model: str = "rerank-english-v3.0"  # Cohere model


@dataclass
class GenerationConfig:
    """LLM Generation Configuration"""
    model_name: str = "gpt-4o"
    temperature: float = 0.1  # Low for factual accuracy
    max_tokens: Optional[int] = 1000
    
    # Quality Checks
    enable_reflection: bool = False  # Self-reflection for quality assurance
    
    # Multimodal
    max_images: int = 5  # Maximum images to include in prompt
    image_encoding: str = "base64"


@dataclass
class SystemConfig:
    """Overall System Configuration"""
    # Sub-configs - use field(default_factory=...) for mutable defaults
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Logging
    log_level: str = "INFO"
    enable_langsmith: bool = False  # LangSmith tracing
    
    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Load API keys from environment"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        
        # Update sub-configs with environment values
        if self.qdrant_api_key:
            self.qdrant.api_key = self.qdrant_api_key
        
        # Use local Qdrant if URL is set
        qdrant_url = os.getenv("QDRANT_URL")
        if qdrant_url:
            self.qdrant.url = qdrant_url


# ============================================================================
# Production Configurations
# ============================================================================

def get_development_config() -> SystemConfig:
    """Configuration for local development"""
    config = SystemConfig()
    config.qdrant.url = ":memory:"  # In-memory for testing
    config.retrieval.enable_reranking = False  # Save Cohere credits
    config.generation.enable_reflection = False
    config.log_level = "DEBUG"
    return config


def get_production_config() -> SystemConfig:
    """Configuration for production deployment"""
    config = SystemConfig()
    
    # Use cloud Qdrant
    config.qdrant.url = os.getenv("QDRANT_URL", "http://localhost:6333")
    config.qdrant.prefer_grpc = True  # Better performance
    
    # Enable advanced features
    config.retrieval.enable_reranking = True
    config.retrieval.enable_hybrid_search = True
    config.generation.enable_reflection = True
    
    # Optimized HNSW parameters for production
    config.qdrant.hnsw_m = 32  # Higher accuracy
    config.qdrant.hnsw_ef_construct = 300
    config.qdrant.hnsw_ef_search = 256
    
    # Enable monitoring
    config.enable_langsmith = True
    
    return config


def get_benchmark_config() -> SystemConfig:
    """Configuration optimized for benchmarking"""
    config = SystemConfig()
    
    # Maximum performance
    config.qdrant.hnsw_m = 64
    config.qdrant.hnsw_ef_construct = 500
    config.qdrant.hnsw_ef_search = 512
    config.qdrant.enable_quantization = False  # Disable for max accuracy
    
    # Large retrieval pool
    config.retrieval.top_k_candidates = 50
    config.retrieval.enable_reranking = True
    
    return config


# ============================================================================
# Helper Functions
# ============================================================================

def load_config(env: str = "development") -> SystemConfig:
    """
    Load configuration based on environment.
    
    Args:
        env: "development", "production", or "benchmark"
    
    Returns:
        SystemConfig instance
    """
    if env == "development":
        return get_development_config()
    elif env == "production":
        return get_production_config()
    elif env == "benchmark":
        return get_benchmark_config()
    else:
        raise ValueError(f"Unknown environment: {env}")


# ============================================================================
# Performance Tuning Guide
# ============================================================================

"""
PERFORMANCE TUNING GUIDE
========================

Latency vs Accuracy Trade-offs:
--------------------------------

1. **For Minimum Latency (<50ms retrieval)**:
   - hnsw_m = 16
   - hnsw_ef_construct = 100
   - hnsw_ef_search = 64
   - enable_quantization = True
   - top_k_candidates = 10

2. **For Maximum Accuracy**:
   - hnsw_m = 64
   - hnsw_ef_construct = 500
   - hnsw_ef_search = 512
   - enable_quantization = False
   - top_k_candidates = 50

3. **Balanced (Recommended)**:
   - hnsw_m = 16-32
   - hnsw_ef_construct = 200
   - hnsw_ef_search = 128
   - enable_quantization = True
   - top_k_candidates = 20

Memory Optimization:
-------------------

- Quantization: Reduces memory by 75% (minimal accuracy loss)
- On-disk storage: Reduces RAM usage for large collections
- Batch processing: Improves throughput

Scaling:
--------

- Single node: Up to 10M vectors, 500-1000 QPS
- Distributed: Horizontal sharding for 100M+ vectors
- Use prefer_grpc=True for 2-3x throughput improvement

Monitoring:
-----------

- Enable LangSmith for LLM call tracing
- Use Qdrant's built-in metrics (Prometheus)
- Track: latency p50/p95/p99, recall@k, precision@k
"""
