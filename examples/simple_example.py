"""
Simple Example: Getting Started with Multimodal RAG
====================================================

This example shows the simplest way to use the system:
1. Initialize RAG
2. Add documents
3. Query

Run: python examples/simple_example.py
"""

from app.rag_orchestrator import create_rag_system
from app.vector_stores.qdrant_store import Document

def main():
    print("🚀 Multimodal RAG - Simple Example\n")
    
    # Step 1: Initialize RAG system (in-memory)
    print("📦 Initializing RAG system...")
    rag = create_rag_system()
    
    # Step 2: Create sample documents
    print("📄 Creating sample documents...")
    documents = [
        Document(
            content="""Python is a high-level, interpreted programming language known for its 
            simplicity and readability. Created by Guido van Rossum and first released in 1991, 
            Python emphasizes code readability with significant whitespace.""",
            modality="text",
            source="python_intro.txt",
            metadata={"category": "programming", "language": "python"}
        ),
        Document(
            content="""Machine Learning is a subset of artificial intelligence that focuses on 
            building systems that can learn from and make decisions based on data. It involves 
            training algorithms on large datasets to identify patterns and make predictions.""",
            modality="text",
            source="ml_basics.txt",
            metadata={"category": "ai", "topic": "machine_learning"}
        ),
        Document(
            content="""Natural Language Processing (NLP) is a branch of AI that helps computers 
            understand, interpret, and generate human language. Applications include chatbots, 
            translation services, and sentiment analysis.""",
            modality="text",
            source="nlp_overview.txt",
            metadata={"category": "ai", "topic": "nlp"}
        ),
        Document(
            content="""The Transformer architecture, introduced in 2017, revolutionized NLP by 
            using self-attention mechanisms instead of recurrence. This enabled parallel processing 
            and led to models like GPT and BERT.""",
            modality="text",
            source="transformers.txt",
            metadata={"category": "ai", "topic": "deep_learning", "year": 2017}
        )
    ]
    
    # Step 3: Embed documents
    print("🧬 Generating CLIP embeddings...")
    embedded_docs = rag.embed_documents(documents, show_progress=True)
    
    # Step 4: Index documents
    print("📊 Indexing into Qdrant...")
    rag.index_documents(embedded_docs)
    
    # Step 5: Show statistics
    stats = rag.get_stats()
    print(f"\n✅ Indexing complete!")
    print(f"   - Total vectors: {stats['total_vectors']}")
    print(f"   - Collection: {stats['collection_name']}")
    print(f"   - Model: {stats['model']}")
    
    # Step 6: Query examples
    print("\n" + "="*80)
    print("QUERYING THE SYSTEM")
    print("="*80)
    
    queries = [
        "What is Python?",
        "Explain machine learning",
        "What year was the Transformer architecture introduced?",
        "Compare NLP and machine learning"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 80)
        
        result = rag.query(query, top_k=3)
        
        print(f"\n{result['answer']}\n")
        print(f"📊 Metadata:")
        print(f"   - Confidence: {result['confidence']}")
        print(f"   - Processing time: {result['metadata']['total_time_ms']:.0f}ms")
        print(f"   - Retrieval: {result['metadata']['retrieval_time_ms']:.0f}ms")
        print(f"   - Generation: {result['metadata']['generation_time_ms']:.0f}ms")
        print(f"   - Sources used: {result['metadata']['num_sources']}")
        
        if result.get('sources'):
            print(f"\n📚 Sources:")
            for source in result['sources']:
                score = source.get('score', 0)
                print(f"   - {source['source']} (relevance: {score:.3f})")
    
    print("\n" + "="*80)
    print("✅ Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
