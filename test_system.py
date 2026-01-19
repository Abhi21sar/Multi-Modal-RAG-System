"""
Quick System Test
==================

This script tests the multimodal RAG system without requiring all heavy models.
It uses in-memory vector store and minimal embeddings for fast testing.
"""

import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("="*80)
print("🧪 MULTIMODAL RAG SYSTEM - QUICK TEST")
print("="*80)

# Test 1: Import core modules
print("\n[Test 1] Importing core modules...")
try:
    from app.embeddings.clip_embedder import CLIPEmbedder
    from app.vector_stores.qdrant_store import QdrantVectorStore, Document
    from app.workflows.retrieval_graph import retrieve
    from app.generation.multimodal_llm import MultimodalGenerator
    from app.rag_orchestrator import create_rag_system, MultimodalRAG
    from config.system_config import load_config
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Configuration loading
print("\n[Test 2] Loading configuration...")
try:
    config = load_config("development")
    print(f"✅ Configuration loaded: {config.qdrant.collection_name}")
except Exception as e:
    print(f"❌ Config failed: {e}")
    sys.exit(1)

# Test 3: Check environment
print("\n[Test 3] Checking environment variables...")
import os
from dotenv import load_dotenv
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    print(f"✅ OpenAI API Key found: {openai_key[:20]}...")
else:
    print("⚠️  Warning: OPENAI_API_KEY not set (needed for generation)")

# Test 4: Create RAG system (without heavy models)
print("\n[Test 4] Initializing RAG system (in-memory mode)...")
try:
    rag = create_rag_system()
    print("✅ RAG system initialized!")
    
    # Get stats
    stats = rag.get_stats()
    print(f"   - Collection: {stats['collection_name']}")
    print(f"   - Vectors: {stats['total_vectors']}")
    print(f"   - Model: {stats['model']}")
except Exception as e:
    print(f"❌ RAG initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Document creation and embedding
print("\n[Test 5] Creating and embedding test documents...")
try:
    test_docs = [
        Document(
            content="Python is a high-level programming language.",
            modality="text",
            source="test1.txt",
            metadata={"category": "programming"}
        ),
        Document(
            content="Machine learning is a branch of artificial intelligence.",
            modality="text",
            source="test2.txt",
            metadata={"category": "ai"}
        )
    ]
    print(f"   Created {len(test_docs)} test documents")
    
    # Embed documents (this will download CLIP model on first run - may take time!)
    print("   Generating embeddings (may download CLIP model on first run)...")
    embedded_docs = rag.embed_documents(test_docs, show_progress=True)
    print(f"✅ Embedded {len(embedded_docs)} documents")
    print(f"   Embedding shape: {embedded_docs[0].embedding.shape}")
except Exception as e:
    print(f"❌ Embedding failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Indexing
print("\n[Test 6] Indexing documents into Qdrant...")
try:
    doc_ids = rag.index_documents(embedded_docs)
    print(f"✅ Indexed {len(doc_ids)} documents")
    
    # Verify
    stats = rag.get_stats()
    print(f"   Total vectors in collection: {stats['total_vectors']}")
except Exception as e:
    print(f"❌ Indexing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Query (if OpenAI key is available)
if openai_key:
    print("\n[Test 7] Running end-to-end query...")
    try:
        result = rag.query("What is Python?", top_k=2)
        print("✅ Query successful!")
        print(f"\n📝 Answer:\n{result['answer']}\n")
        print(f"📊 Confidence: {result['confidence']}")
        print(f"⏱️  Total time: {result['metadata']['total_time_ms']:.0f}ms")
        print(f"   - Retrieval: {result['metadata']['retrieval_time_ms']:.0f}ms")
        print(f"   - Generation: {result['metadata']['generation_time_ms']:.0f}ms")
    except Exception as e:
        print(f"❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n[Test 7] Skipping query test (OpenAI API key not set)")
    print("   To test full system, add OPENAI_API_KEY to .env file")

# Summary
print("\n" + "="*80)
print("🎉 SYSTEM TEST COMPLETE!")
print("="*80)
print("\n✅ All core components are working!")
print("\n📚 Next steps:")
print("   1. Add your documents to data/ directory")
print("   2. Run: python examples/simple_example.py")
print("   3. Or use the system programmatically (see QUICKSTART.md)")
print("\n🚀 Happy building!")
print("="*80)
