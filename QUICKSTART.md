# 🚀 Quick Start Guide - Multimodal RAG System

Get your production-grade multimodal RAG system running in **5 minutes**!

---

## 📋 Prerequisites

- Python 3.11 or 3.12
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- (Optional) Docker for local Qdrant instance
- (Optional) Cohere API key for reranking

---

## ⚡ Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -r requirements.txt
```

### Step 2: Configure Environment

The `.env` file is already configured with in-memory storage. Just ensure your **OpenAI API key** is set:

```bash
# .env file already contains:
OPENAI_API_KEY=your-key-here
QDRANT_URL=:memory:  # In-memory for testing
```

### Step 3: Run Your First Query!

```python
from app.rag_orchestrator import create_rag_system
from app.vector_stores.qdrant_store import Document

# Initialize RAG system
rag = create_rag_system()

# Create sample documents
docs = [
    Document(
        content="Python is a high-level programming language known for readability.",
        modality="text",
        source="python_guide.txt"
    ),
    Document(
        content="Machine learning is a subset of AI focused on learning from data.",
        modality="text",
        source="ml_intro.txt"
    )
]

# Embed and index
embedded_docs = rag.embed_documents(docs)
rag.index_documents(embedded_docs)

# Query!
result = rag.query("What is Python?")
print(result["answer"])
```

**Output:**
```
Python is a high-level programming language known for its readability. [Source 1]

Confidence: High
```

---

## 🐳 Using Qdrant with Docker (Recommended for Persistence)

### Start Qdrant Server

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### Update .env

```bash
QDRANT_URL=http://localhost:6333
```

Now your vectors will persist across sessions!

---

## 📁 Complete Example: Index Your Documents

Create a file `index_my_data.py`:

```python
"""
Index your documents into the multimodal RAG system
"""

from pathlib import Path
from app.rag_orchestrator import create_rag_system
from app.vector_stores.qdrant_store import Document
from app.ingestion.ingestion import extract_text_from_pdf  # We'll create this

# Initialize RAG
rag = create_rag_system()

# Load documents from data/ directory
data_dir = Path("data")
documents = []

# Index PDFs
for pdf_file in data_dir.glob("*.pdf"):
    text = extract_text_from_pdf(pdf_file)  # From existing ingestion module
    doc = Document(
        content=text,
        modality="text",
        source=str(pdf_file),
        metadata={"type": "pdf", "filename": pdf_file.name}
    )
    documents.append(doc)

# Index text files
for txt_file in data_dir.glob("*.txt"):
    with open(txt_file) as f:
        text = f.read()
    doc = Document(
        content=text,
        modality="text",
        source=str(txt_file),
        metadata={"type": "text", "filename": txt_file.name}
    )
    documents.append(doc)

# Index images (using CLIP vision encoder)
for img_file in data_dir.glob("*.{jpg,png,jpeg}"):
    # For images, content is the caption/description (if available)
    # Or we can use OCR
    doc = Document(
        content=f"Image: {img_file.name}",  # Placeholder
        modality="image",
        source=str(img_file),
        metadata={"type": "image", "filename": img_file.name}
    )
    documents.append(doc)

print(f"Found {len(documents)} documents")

# Embed and index
print("Generating embeddings...")
embedded_docs = rag.embed_documents(documents, show_progress=True)

print("Indexing into Qdrant...")
rag.index_documents(embedded_docs)

print("✅ Indexing complete!")
print(f"\nStats: {rag.get_stats()}")
```

### Run Indexing

```bash
python index_my_data.py
```

---

## 💬 Query Your Data

Create `query_system.py`:

```python
from app.rag_orchestrator import create_rag_system

# Initialize (will load existing Qdrant collection)
rag = create_rag_system()

# Interactive query loop
while True:
    query = input("\n🔍 Ask a question (or 'quit'): ")
    if query.lower() == 'quit':
        break
    
    result = rag.query(query, top_k=5)
    
    print(f"\n✅ Answer:\n{result['answer']}\n")
    print(f"Confidence: {result['confidence']}")
    print(f"Processing time: {result['metadata']['total_time_ms']:.0f}ms")
    
    print(f"\nSources:")
    for source in result.get('sources', []):
        print(f"  - {source['source']} (score: {source.get('score', 0):.3f})")
```

### Run Query Interface

```bash
python query_system.py
```

---

## 🎨 Using the Streamlit UI

We'll update the existing `ui.py`:

```bash
streamlit run ui.py
```

This will open a web interface where you can:
- Upload documents
- Ask questions
- View sources with citations
- See confidence scores

---

## 🔧 Advanced Configuration

### Enable Reranking (Improves Precision by 20-30%)

1. Get Cohere API key: https://dashboard.cohere.com/api-keys
2. Update `.env`:
   ```bash
   COHERE_API_KEY=your-cohere-key
   ```
3. Enable in code:
   ```python
   rag = create_rag_system(enable_reranking=True)
   ```

### Enable Self-Reflection (Quality Checks)

```python
rag = create_rag_system(enable_reflection=True)
```

This uses a second LLM call to verify answer quality.

### Use Production Configuration

```python
from config.system_config import load_config
from app.rag_orchestrator import MultimodalRAG

config = load_config("production")
rag = MultimodalRAG(
    qdrant_url=config.qdrant.url,
    model_name=config.generation.model_name,
    enable_reranking=config.retrieval.enable_reranking,
    enable_reflection=config.generation.enable_reflection
)
```

---

## 📊 Monitoring & Observability

### LangSmith (LLM Tracing)

1. Get API key: https://smith.langchain.com/
2. Update `.env`:
   ```bash
   LANGSMITH_API_KEY=your-key
   LANGCHAIN_TRACING_V2=true
   ```

Now all LLM calls will be traced in the LangSmith dashboard!

### Qdrant Metrics

Access Qdrant dashboard at: http://localhost:6333/dashboard

View:
- Collection statistics
- Query latency
- Memory usage
- Index status

---

## 🚀 Performance Benchmarking

Run the benchmark script:

```bash
python scripts/benchmark.py
```

This will test:
- Retrieval latency (p50, p95, p99)
- End-to-end query time
- Recall@k accuracy
- Memory usage

---

## 📝 Example Use Cases

### 1. Research Assistant

```python
result = rag.query(
    "Compare the revenue growth in Q1 vs Q2",
    top_k=10  # Retrieve more for comparison queries
)
```

### 2. Document QA

```python
result = rag.query("What are the key findings in the study?")
```

### 3. Image Search

```python
# Query for images using text
result = rag.query("Show me diagrams about neural networks")
# Will retrieve images embedded with CLIP
```

### 4. Multimodal Analysis

```python
# The system automatically handles both text and images
result = rag.query("Explain the chart on page 5")
```

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"

**Solution:** CLIP embedder will auto-detect CPU. To force CPU:
```python
from app.embeddings.clip_embedder import CLIPEmbedder
embedder = CLIPEmbedder(device="cpu")
```

### Issue: "Qdrant connection refused"

**Solution:** Ensure Docker is running:
```bash
docker ps  # Check if qdrant container is running
docker run -p 6333:6333 qdrant/qdrant  # Start if not running
```

### Issue: Slow queries

**Solution:** Check Qdrant is using quantization:
```python
# In system_config.py
config.qdrant.enable_quantization = True
config.qdrant.hnsw_ef_search = 64  # Lower = faster
```

---

## 📚 Next Steps

1. **Read the Architecture**: `ARCHITECTURE.md`
2. **Implementation Details**: `IMPLEMENTATION_PLAN.md`
3. **Deploy to Production**: See `docs/deployment.md` (coming soon)
4. **Customize Prompts**: Edit `app/generation/multimodal_llm.py`
5. **Add Custom Ingestion**: Extend `app/ingestion/`

---

## 💡 Pro Tips

1. **Start with in-memory Qdrant** (`:memory:`) for testing
2. **Use Docker Qdrant** for persistence
3. **Enable reranking** for production (worth the extra latency)
4. **Monitor with LangSmith** to debug prompts
5. **Batch index documents** (100-1000 at a time) for speed

---

## 🤝 Need Help?

- **Issues**: Open a GitHub issue
- **Questions**: Check `docs/FAQ.md`
- **Examples**: See `examples/` directory

**Happy building! 🚀**
