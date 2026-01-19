"""
Multi-Modal Embedder using CLIP
=================================

This module provides a unified embedding interface for both text and images using OpenAI's CLIP model.
CLIP maps text and images to the same 768-dimensional vector space, enabling cross-modal semantic search.

Key Features:
- ViT-L/14@336px architecture (768-dim embeddings)
- L2-normalized vectors for cosine similarity
- Batch processing support
- GPU acceleration when available
- Consistent embedding space for text and images

Author: Abhishek Gurjar
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Union
from transformers import CLIPProcessor, CLIPModel
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """
    Unified embedder for text and images using CLIP.
    
    This class ensures that both text and images are embedded into the same
    vector space, which is critical for multimodal retrieval.
    """
    
    def __init__(
        self, 
        model_name: str = "openai/clip-vit-large-patch14-336",
        device: str = None
    ):
        """
        Initialize CLIP model and processor.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading CLIP model: {model_name} on device: {self.device}")
        
        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Set to evaluation mode
        self.model.eval()
        
        logger.info(f"CLIP model loaded successfully. Embedding dimension: 768")
    
    def embed_text(self, text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Embed text into 768-dimensional vector space.
        
        Args:
            text: Single text string or list of strings
            normalize: Whether to L2-normalize vectors (recommended for cosine similarity)
        
        Returns:
            numpy array of shape (768,) for single text or (n, 768) for batch
        """
        # Ensure text is a list
        if isinstance(text, str):
            text = [text]
            single_input = True
        else:
            single_input = False
        
        # Process text
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77  # CLIP's max token length
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        # Convert to numpy
        embeddings = text_features.cpu().numpy()
        
        # Normalize if requested
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Return single vector if single input
        return embeddings[0] if single_input else embeddings
    
    def embed_image(
        self, 
        image: Union[Image.Image, str, Path, List[Union[Image.Image, str, Path]]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Embed image(s) into 768-dimensional vector space (same space as text).
        
        Args:
            image: PIL Image, path to image, or list of images/paths
            normalize: Whether to L2-normalize vectors
        
        Returns:
            numpy array of shape (768,) for single image or (n, 768) for batch
        """
        # Handle single image vs batch
        if not isinstance(image, list):
            images = [image]
            single_input = True
        else:
            images = image
            single_input = False
        
        # Load images if paths are provided
        loaded_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                loaded_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                loaded_images.append(img.convert("RGB"))
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        
        # Process images
        inputs = self.processor(
            images=loaded_images,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        # Convert to numpy
        embeddings = image_features.cpu().numpy()
        
        # Normalize if requested
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings[0] if single_input else embeddings
    
    def compute_similarity(
        self, 
        text_embedding: np.ndarray, 
        image_embedding: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between text and image embeddings.
        
        Args:
            text_embedding: 768-dim text vector
            image_embedding: 768-dim image vector
        
        Returns:
            Similarity score between -1 and 1 (higher is more similar)
        """
        # Ensure vectors are normalized
        text_norm = text_embedding / np.linalg.norm(text_embedding)
        image_norm = image_embedding / np.linalg.norm(image_embedding)
        
        return float(np.dot(text_norm, image_norm))
    
    def batch_embed_texts(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Efficiently embed large batches of text.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
        
        Returns:
            numpy array of shape (n, 768)
        """
        all_embeddings = []
        
        # Import tqdm if progress bar requested
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(texts), batch_size), desc="Embedding texts")
        else:
            iterator = range(0, len(texts), batch_size)
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            embeddings = self.embed_text(batch, normalize=True)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def batch_embed_images(
        self, 
        image_paths: List[Union[str, Path]], 
        batch_size: int = 16,  # Smaller batch for images (memory intensive)
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Efficiently embed large batches of images.
        
        Args:
            image_paths: List of paths to images
            batch_size: Number of images to process at once
            show_progress: Whether to show progress bar
        
        Returns:
            numpy array of shape (n, 768)
        """
        all_embeddings = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(image_paths), batch_size), desc="Embedding images")
        else:
            iterator = range(0, len(image_paths), batch_size)
        
        for i in iterator:
            batch = image_paths[i:i + batch_size]
            embeddings = self.embed_image(batch, normalize=True)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension (768 for CLIP-ViT-L/14)."""
        return 768


# Example usage and testing
if __name__ == "__main__":
    # Initialize embedder
    embedder = CLIPEmbedder()
    
    # Test text embedding
    text = "A photo of a dog playing in the park"
    text_embedding = embedder.embed_text(text)
    print(f"Text embedding shape: {text_embedding.shape}")
    print(f"Text embedding (first 5 dims): {text_embedding[:5]}")
    
    # Test batch text embedding
    texts = ["A cat sleeping", "A dog running", "A bird flying"]
    batch_embeddings = embedder.embed_text(texts)
    print(f"\nBatch embeddings shape: {batch_embeddings.shape}")
    
    # Test similarity computation
    text1 = embedder.embed_text("A dog")
    text2 = embedder.embed_text("A puppy")
    text3 = embedder.embed_text("A car")
    
    print(f"\nSimilarity between 'dog' and 'puppy': {embedder.compute_similarity(text1, text2):.4f}")
    print(f"Similarity between 'dog' and 'car': {embedder.compute_similarity(text1, text3):.4f}")
    
    logger.info("CLIP Embedder test completed successfully!")
