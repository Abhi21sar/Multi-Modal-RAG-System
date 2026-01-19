"""
Batch Document Indexing Script
================================

Index large volumes of documents (PDFs, text files, images) into the RAG system.
Supports parallel processing, progress tracking, and error recovery.

Usage:
    poetry run python scripts/index_documents.py --data-dir data/ --batch-size 100

Author: Abhishek Gurjar
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from typing import List
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag_orchestrator import create_rag_system
from app.vector_stores.qdrant_store import Document
from app.ingestion import extract_text_from_pdf, extract_text_from_image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
    """
    Chunk text into smaller pieces with overlap.
    
    Args:
        text: Input text
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < text_len:
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size // 2:  # Only break if we're past halfway
                end = start + break_point + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= text_len:
            break
    
    return [c for c in chunks if c]  # Filter empty chunks


def process_pdf(file_path: Path, source_name: str) -> List[Document]:
    """Process a PDF file into document chunks."""
    try:
        logger.info(f"Processing PDF: {source_name}")
        text = extract_text_from_pdf(str(file_path))
        
        if not text or len(text) < 50:
            logger.warning(f"PDF {source_name} has insufficient text")
            return []
        
        # Chunk the text
        chunks = chunk_text(text, chunk_size=512, overlap=128)
        
        # Create documents
        docs = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                content=chunk,
                modality="text",
                source=source_name,
                metadata={
                    "type": "pdf",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_path": str(file_path)
                }
            )
            docs.append(doc)
        
        logger.info(f"  Created {len(docs)} chunks from {source_name}")
        return docs
    
    except Exception as e:
        logger.error(f"Error processing PDF {source_name}: {e}")
        return []


def process_text_file(file_path: Path, source_name: str) -> List[Document]:
    """Process a text file into document chunks."""
    try:
        logger.info(f"Processing text file: {source_name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if not text:
            logger.warning(f"Text file {source_name} is empty")
            return []
        
        # Chunk the text
        chunks = chunk_text(text, chunk_size=512, overlap=128)
        
        # Create documents
        docs = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                content=chunk,
                modality="text",
                source=source_name,
                metadata={
                    "type": "text",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_path": str(file_path)
                }
            )
            docs.append(doc)
        
        logger.info(f"  Created {len(docs)} chunks from {source_name}")
        return docs
    
    except Exception as e:
        logger.error(f"Error processing text file {source_name}: {e}")
        return []


def process_image(file_path: Path, source_name: str) -> List[Document]:
    """Process an image file (OCR + visual embedding)."""
    try:
        logger.info(f"Processing image: {source_name}")
        # Extract text via OCR
        text = extract_text_from_image(str(file_path))
        
        # Create document with both text and image reference
        doc = Document(
            content=text if text else f"Image: {source_name}",
            modality="image",
            source=source_name,
            metadata={
                "type": "image",
                "file_path": str(file_path),
                "has_ocr_text": bool(text)
            }
        )
        
        logger.info(f"  Created 1 image document from {source_name}")
        return [doc]
    
    except Exception as e:
        logger.error(f"Error processing image {source_name}: {e}")
        return []


def index_directory(
    data_dir: str,
    use_cloud: bool = False,
    enable_reranking: bool = False,
    batch_size: int = 100,
    file_types: List[str] = None
):
    """
    Index all documents in a directory.
    
    Args:
        data_dir: Directory containing documents
        use_cloud: Use Qdrant Cloud instead of local
        enable_reranking: Enable Cohere reranking
        batch_size: Batch size for indexing
        file_types: List of file extensions to process (default: all)
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Initialize RAG system
    logger.info("="*80)
    logger.info("INITIALIZING RAG SYSTEM")
    logger.info("="*80)
    
    qdrant_url = ":memory:" if not use_cloud else os.getenv("QDRANT_URL")
    
    rag = create_rag_system(
        use_cloud=use_cloud,
        enable_reranking=enable_reranking
    )
    
    # Define file type handlers
    if file_types is None:
        file_types = ['.pdf', '.txt', '.md', '.jpg', '.jpeg', '.png']
    
    handlers = {
        '.pdf': process_pdf,
        '.txt': process_text_file,
        '.md': process_text_file,
        '.jpg': process_image,
        '.jpeg': process_image,
        '.png': process_image,
    }
    
    # Collect all files
    logger.info("="*80)
    logger.info("SCANNING DIRECTORY")
    logger.info("="*80)
    
    all_files = []
    for ext in file_types:
        files = list(data_path.rglob(f"*{ext}"))
        all_files.extend(files)
        logger.info(f"Found {len(files)} {ext} files")
    
    logger.info(f"\nTotal files to process: {len(all_files)}")
    
    if not all_files:
        logger.warning("No files found to index!")
        return
    
    # Process files
    logger.info("\n" + "="*80)
    logger.info("PROCESSING FILES")
    logger.info("="*80 + "\n")
    
    all_documents = []
    
    for file_path in tqdm(all_files, desc="Processing files"):
        ext = file_path.suffix.lower()
        handler = handlers.get(ext)
        
        if handler:
            docs = handler(file_path, file_path.name)
            all_documents.extend(docs)
    
    logger.info(f"\n✅ Processed {len(all_files)} files → {len(all_documents)} document chunks")
    
    if not all_documents:
        logger.warning("No documents created!")
        return
    
    # Embed documents
    logger.info("\n" + "="*80)
    logger.info("GENERATING EMBEDDINGS")
    logger.info("="*80 + "\n")
    
    embedded_docs = rag.embed_documents(all_documents, show_progress=True)
    
    # Index documents
    logger.info("\n" + "="*80)
    logger.info("INDEXING INTO VECTOR STORE")
    logger.info("="*80 + "\n")
    
    doc_ids = rag.index_documents(embedded_docs, batch_size=batch_size)
    
    # Show statistics
    logger.info("\n" + "="*80)
    logger.info("INDEXING COMPLETE!")
    logger.info("="*80)
    
    stats = rag.get_stats()
    logger.info(f"\n📊 Final Statistics:")
    logger.info(f"   - Files processed: {len(all_files)}")
    logger.info(f"   - Documents created: {len(all_documents)}")
    logger.info(f"   - Vectors indexed: {stats['total_vectors']}")
    logger.info(f"   - Collection: {stats['collection_name']}")
    logger.info(f"   - Model: {stats['model']}")
    
    # File type breakdown
    logger.info(f"\n📁 File Type Breakdown:")
    type_counts = {}
    for doc in all_documents:
        doc_type = doc.metadata.get('type', 'unknown')
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    
    for doc_type, count in sorted(type_counts.items()):
        logger.info(f"   - {doc_type}: {count} chunks")
    
    logger.info("\n✅ Document indexing complete!")
    logger.info("   You can now query your documents using the RAG system.\n")


def main():
    parser = argparse.ArgumentParser(description="Index documents into RAG system")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing documents to index'
    )
    parser.add_argument(
        '--use-cloud',
        action='store_true',
        help='Use Qdrant Cloud instead of local storage'
    )
    parser.add_argument(
        '--enable-reranking',
        action='store_true',
        help='Enable Cohere reranking'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for indexing'
    )
    parser.add_argument(
        '--file-types',
        type=str,
        nargs='+',
        help='File extensions to process (e.g., .pdf .txt)'
    )
    
    args = parser.parse_args()
    
    index_directory(
        data_dir=args.data_dir,
        use_cloud=args.use_cloud,
        enable_reranking=args.enable_reranking,
        batch_size=args.batch_size,
        file_types=args.file_types
    )


if __name__ == "__main__":
    main()
