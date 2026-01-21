import streamlit as st
import sys
import os
import time
from pathlib import Path
from PIL import Image
import base64

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.rag_orchestrator import create_rag_system
from app.vector_stores.qdrant_store import Document

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Multimodal RAG Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1e1e2f, #121212);
        color: #e0e0e0;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 600;
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 2rem;
    }
    
    .source-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .source-card:hover {
        transform: translateY(-5px);
        border-color: #2575fc;
        background: rgba(37, 117, 252, 0.05);
    }
    
    .confidence-high { color: #00ff88; font-weight: 600; }
    .confidence-medium { color: #ffbb00; font-weight: 600; }
    .confidence-low { color: #ff4444; font-weight: 600; }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 100px !important;
        padding: 0.75rem 1.5rem !important;
    }
    
    .stButton > button {
        border-radius: 100px !important;
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%) !important;
        border: none !important;
        color: white !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# State Management & System Initialization
# =============================================================================
@st.cache_resource
def get_rag():
    # Using production settings by default in the UI for better quality
    return create_rag_system(
        use_cloud=False,  # Set to True if using Qdrant Cloud
        enable_reranking=os.getenv("COHERE_API_KEY") is not None,
        enable_reflection=True
    )

rag = get_rag()

if 'history' not in st.session_state:
    st.session_state.history = []

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.image("multimodal_rag_architecture_1768848784409.png", caption="System Architecture", use_container_width=True)
    st.markdown("---")
    st.markdown("### ⚙️ System Configuration")
    precision_boost = st.toggle("Enable Precision Reranking", value=True)
    reflection_mode = st.toggle("Enable Quality Reflection", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 Collection Stats")
    stats = rag.get_stats()
    st.metric("Total Documents", stats.get("total_vectors", 0))
    st.metric("Status", stats.get("status", "Unknown").upper())
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# =============================================================================
# Main UI
# =============================================================================
st.markdown('<h1 class="main-header">🧠 Intelligence Bot</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Multimodal Retrieval Augmented Generation</p>', unsafe_allow_html=True)

# Search Interface
query = st.text_input("Search across text, images, and documents...", placeholder="e.g., What is our strategy for Q3? Show me diagrams of neural networks.")

if query:
    with st.spinner("Processing multimodal retrieval and generating answer..."):
        start_time = time.time()
        
        # Execute query
        try:
            result = rag.query(
                query, 
                top_k=5, 
                return_sources=True, 
                return_context=True
            )
            
            elapsed_time = time.time() - start_time
            
            # 1. Answer Section
            st.markdown("### 🤖 Answer")
            
            # Confidence Indicator
            conf = result.get('confidence', 'Medium')
            conf_class = f"confidence-{conf.lower()}"
            st.markdown(f"**Confidence Level**: <span class='{conf_class}'>{conf}</span>", unsafe_allow_html=True)
            
            # Display answer with typewriter effect (optional, here just markdown)
            st.write(result['answer'])
            
            # 2. Results Tabs
            st.markdown("---")
            res_tab, src_tab, raw_tab = st.tabs(["🖼️ Visual Results", "📚 Citations", "🛠️ Metadata"])
            
            with res_tab:
                # Filter results that are images
                images_found = [s for s in result.get('sources', []) if s.get('modality') == 'image']
                
                if images_found:
                    cols = st.columns(min(len(images_found), 3))
                    for idx, img_src in enumerate(images_found):
                        with cols[idx % 3]:
                            try:
                                # In a real scenario, this would be a path or URL
                                path = img_src.get('source')
                                if os.path.exists(path):
                                    st.image(path, caption=f"Source [{img_src['index']}]", use_container_width=True)
                                else:
                                    st.info(f"Image reference found at {path}")
                            except Exception as e:
                                st.error(f"Error loading image: {e}")
                else:
                    st.info("No visual documents were relevant to this query.")
            
            with src_tab:
                # Display text sources
                for src in result.get('sources', []):
                    with st.container():
                        st.markdown(f"""
                        <div class="source-card">
                            <b>[Source {src['index']}]: {src['source']}</b><br>
                            <small>Modality: {src['modality']} | Relevance: {src['score']:.4f}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        source_id = f"content_{src['index']}"
                        with st.expander("View Source Snippet"):
                            # Find original content in result context or fetch from vector store logic
                            # (Simplified here)
                            st.write(result.get('context', '').split('---')[src['index']-1] if '---' in result.get('context', '') else "Context snippet available in metadata.")

            with raw_tab:
                st.json(result['metadata'])
                st.info(f"End-to-end processing time: {elapsed_time:.2f}s")

        except Exception as e:
            st.error(f"System Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Welcome Message if no query
else:
    st.markdown("""
    ### Welcome to the Multimodal RAG Explorer!
    
    This system uses **CLIP** embeddings to understand both text and images in a single semantic space. 
    You can query the knowledge base and it will retrieve the most relevant information regardless of whether it's stored as text or images.
    
    **Try these common queries:**
    - "Can you summarize the main architecture?"
    - "Find diagrams related to the retrieval pipeline."
    - "What are the performance metrics of the system?"
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #555;'><small>Built with LangGraph, Qdrant, and GPT-4o</small></div>", unsafe_allow_html=True)