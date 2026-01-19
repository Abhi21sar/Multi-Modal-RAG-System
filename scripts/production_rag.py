"""
Production RAG System with Advanced Features
=============================================

This script demonstrates enabling all advanced features:
- Cohere reranking (20-30% precision boost)
- Self-reflection (quality checks)
- LangSmith tracing (monitoring)
- Qdrant Cloud (persistence)

Usage:
    PYTHONPATH=. poetry run python scripts/production_rag.py

Author: Abhishek Gurjar
"""

import sys
import os
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag_orchestrator import MultimodalRAG
from config.system_config import load_config
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_production_rag():
    """
    Create RAG system with all advanced features enabled.
    
    Features:
    - Cohere reranking for better precision
    - Self-reflection for quality assurance
    - LangSmith tracing for monitoring
    - Qdrant Cloud for scalability
    """
    logger.info("="*80)
    logger.info("INITIALIZING PRODUCTION RAG SYSTEM")
    logger.info("="*80)
    
    # Load production config
    config = load_config("production")
    
    # Check required API keys
    required_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Cohere": os.getenv("COHERE_API_KEY"),
        "Qdrant": os.getenv("QDRANT_URL"),
    }
    
    logger.info("\n📋 Checking API Keys:")
    for name, key in required_keys.items():
        if key:
            logger.info(f"   ✅ {name}: Configured")
        else:
            logger.warning(f"   ⚠️  {name}: Not configured")
    
    # Enable LangSmith tracing if configured
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    if langsmith_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "multimodal-rag-production"
        logger.info("   ✅ LangSmith: Tracing enabled")
    else:
        logger.info("   ℹ️  LangSmith: Not configured (optional)")
    
    # Initialize RAG with all features
    logger.info("\n🚀 Initializing with advanced features:")
    logger.info(f"   - Qdrant: {config.qdrant.url}")
    logger.info(f"   - HNSW: m={config.qdrant.hnsw_m}, ef={config.qdrant.hnsw_ef_search}")
    logger.info(f"   - Quantization: {config.qdrant.enable_quantization}")
    logger.info(f"   - Reranking: {config.retrieval.enable_reranking}")
    logger.info(f"   - Self-Reflection: {config.generation.enable_reflection}")
    logger.info(f"   - Model: {config.generation.model_name}")
    
    rag = MultimodalRAG(
        collection_name=config.qdrant.collection_name,
        qdrant_url=config.qdrant.url,
        qdrant_api_key=config.qdrant.api_key,
        model_name=config.generation.model_name,
        enable_reranking=config.retrieval.enable_reranking,
        enable_reflection=config.generation.enable_reflection
    )
    
    logger.info("\n✅ Production RAG system initialized!")
    
    return rag


def demo_query(rag, query: str):
    """Run a demo query with full observability."""
    logger.info("\n" + "="*80)
    logger.info(f"QUERY: {query}")
    logger.info("="*80)
    
    result = rag.query(query, top_k=5, return_sources=True)
    
    logger.info(f"\n📝 Answer:")
    logger.info(f"{result['answer']}\n")
    
    logger.info(f"📊 Metadata:")
    logger.info(f"   - Confidence: {result['confidence']}")
    logger.info(f"   - Total time: {result['metadata']['total_time_ms']:.0f}ms")
    logger.info(f"   - Retrieval: {result['metadata']['retrieval_time_ms']:.0f}ms")
    logger.info(f"   - Generation: {result['metadata']['generation_time_ms']:.0f}ms")
    logger.info(f"   - Sources: {result['metadata']['num_sources']}")
    
    if 'quality_score' in result:
        logger.info(f"   - Quality Score: {result['quality_score']}/10")
    
    if result.get('sources'):
        logger.info(f"\n📚 Sources:")
        for source in result['sources']:
            logger.info(f"   - {source['source']} (score: {source.get('score', 0):.3f})")
    
    return result


def main():
    """Main execution function."""
    # Create production RAG
    rag = create_production_rag()
    
    # Get system stats
    stats = rag.get_stats()
    logger.info("\n" + "="*80)
    logger.info("SYSTEM STATISTICS")
    logger.info("="*80)
    logger.info(f"   - Collection: {stats['collection_name']}")
    logger.info(f"   - Total Vectors: {stats['total_vectors']}")
    logger.info(f"   - Indexed Vectors: {stats['indexed_vectors']}")
    logger.info(f"   - Model: {stats['model']}")
    logger.info(f"   - Status: {stats['status']}")
    
    # Demo queries (if collection has data)
    if stats['total_vectors'] > 0:
        logger.info("\n" + "="*80)
        logger.info("RUNNING DEMO QUERIES")
        logger.info("="*80)
        
        demo_queries = [
            "What is the main topic of the documents?",
            "Summarize the key findings",
        ]
        
        for query in demo_queries:
            demo_query(rag, query)
    else:
        logger.info("\n⚠️  Collection is empty. Index documents first:")
        logger.info("   PYTHONPATH=. poetry run python scripts/index_documents.py")
    
    logger.info("\n" + "="*80)
    logger.info("✅ PRODUCTION RAG SYSTEM READY")
    logger.info("="*80)
    logger.info("\n💡 Next steps:")
    logger.info("   1. Index documents: python scripts/index_documents.py")
    logger.info("   2. Query the system: Use the rag.query() method")
    logger.info("   3. Monitor in LangSmith: https://smith.langchain.com/")
    logger.info("   4. Check Qdrant dashboard: http://localhost:6333/dashboard")


if __name__ == "__main__":
    main()
