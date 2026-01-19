"""
Multimodal Generation Pipeline
================================

LangChain-based generation module supporting:
- GPT-4o and Claude 3.5 Sonnet for multimodal reasoning
- Structured prompts with source attribution
- Self-reflection and confidence scoring
- Image-aware generation (base64 encoding)

Author: Abhishek Gurjar
"""

import os
import base64
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Prompt Templates
# ============================================================================

RAG_SYSTEM_PROMPT = """You are an expert research assistant with access to a multimodal knowledge base.

Your task is to answer questions using ONLY the provided context. Follow these guidelines:

1. **Accuracy**: Base your answer strictly on the context. Do not hallucinate or add external knowledge.
2. **Citations**: Always cite sources using the format [Source X] where X is the source number.
3. **Transparency**: If the context doesn't contain enough information, explicitly state this.
4. **Clarity**: Provide clear, well-structured answers.
5. **Confidence**: End your answer with a confidence level (High/Medium/Low) based on context quality.

If the context includes images, reference them appropriately in your answer.
"""

RAG_USER_TEMPLATE = """Context Documents:
{context}

Question: {query}

Instructions:
- Answer the question using the context above
- Cite sources: [Source 1], [Source 2], etc.
- If insufficient information, say "I don't have enough information to answer this fully"
- Indicate your confidence level at the end

Answer:"""


SELF_REFLECTION_TEMPLATE = """Review the following answer and assess its quality:

Original Question: {query}
Context: {context}
Generated Answer: {answer}

Evaluate:
1. Does the answer directly address the question?
2. Are all claims supported by the context?
3. Are citations accurate?
4. Is there any hallucination or external knowledge?

Provide:
- Issues found (if any)
- Suggested improvements
- Overall quality score (0-10)

Evaluation:"""


# ============================================================================
# Structured Output Models
# ============================================================================

class RAGResponse(BaseModel):
    """Structured RAG response with metadata."""
    answer: str = Field(description="The generated answer")
    confidence: str = Field(description="Confidence level: High, Medium, or Low")
    sources_used: List[int] = Field(description="List of source indices cited")
    reasoning: Optional[str] = Field(default="", description="Chain-of-thought reasoning (if applicable)")
    

class ReflectionResult(BaseModel):
    """Self-reflection evaluation result."""
    issues: List[str] = Field(description="List of issues found")
    improvements: List[str] = Field(description="Suggested improvements")
    quality_score: int = Field(description="Quality score from 0-10")


# ============================================================================
# Multimodal Generator
# ============================================================================

class MultimodalGenerator:
    """
    Multimodal generation pipeline using GPT-4o or Claude 3.5 Sonnet.
    
    Features:
    - Text and image-aware generation
    - Source attribution
    - Self-reflection for quality assurance
    - Confidence scoring
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        enable_reflection: bool = False
    ):
        """
        Initialize the multimodal generator.
        
        Args:
            model_name: Model to use ("gpt-4o", "gpt-4-turbo", "claude-3-5-sonnet")
            temperature: Generation temperature (0-1, lower = more deterministic)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            enable_reflection: Enable self-reflection for quality checks
        """
        self.model_name = model_name
        self.temperature = temperature
        self.enable_reflection = enable_reflection
        
        # Initialize LLM
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("user", RAG_USER_TEMPLATE)
        ])
        
        # Create chain
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        logger.info(f"Initialized {model_name} generator (temp={temperature}, reflection={enable_reflection})")
    
    def generate(
        self,
        query: str,
        context: str,
        sources: List[Dict],
        include_images: bool = False
    ) -> Dict[str, any]:
        """
        Generate answer from query and retrieved context.
        
        Args:
            query: User question
            context: Retrieved context with sources
            sources: List of source metadata
            include_images: Whether to include images in the prompt (for multimodal models)
        
        Returns:
            Dictionary with answer, confidence, sources, and metadata
        """
        import time
        start_time = time.time()
        
        # Generate answer
        try:
            answer = self.chain.invoke({
                "query": query,
                "context": context
            })
            
            # Extract confidence level (assumes model follows instructions)
            confidence = self._extract_confidence(answer)
            
            # Extract cited sources
            sources_used = self._extract_cited_sources(answer)
            
            # Self-reflection (optional)
            reflection = None
            quality_score = None
            if self.enable_reflection:
                reflection = self._self_reflect(query, context, answer)
                quality_score = reflection.get("quality_score", None)
            
            generation_time = (time.time() - start_time) * 1000
            
            logger.info(f"Generated answer in {generation_time:.2f}ms (confidence: {confidence})")
            
            return {
                "answer": answer,
                "confidence": confidence,
                "sources_used": sources_used,
                "num_sources": len(sources),
                "generation_time_ms": generation_time,
                "reflection": reflection,
                "quality_score": quality_score
            }
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "confidence": "Low",
                "sources_used": [],
                "error": str(e)
            }
    
    def generate_with_images(
        self,
        query: str,
        context: str,
        sources: List[Dict],
        image_paths: List[str]
    ) -> Dict[str, any]:
        """
        Generate answer with image context (for GPT-4o vision capabilities).
        
        Args:
            query: User question
            context: Text context
            sources: Source metadata
            image_paths: List of paths to relevant images
        
        Returns:
            Generation result with answer and metadata
        """
        # Encode images to base64
        encoded_images = []
        for img_path in image_paths[:5]:  # Limit to 5 images to avoid token limits
            try:
                with open(img_path, "rb") as img_file:
                    encoded = base64.b64encode(img_file.read()).decode("utf-8")
                    encoded_images.append(encoded)
            except Exception as e:
                logger.warning(f"Failed to encode image {img_path}: {e}")
        
        # Build multimodal messages
        content = [
            {"type": "text", "text": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context and images:"}
        ]
        
        for img_b64 in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        
        # Use raw LLM call (bypass chain for multimodal content)
        try:
            response = self.llm.invoke([
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ])
            
            answer = response.content
            confidence = self._extract_confidence(answer)
            sources_used = self._extract_cited_sources(answer)
            
            logger.info(f"Generated multimodal answer with {len(encoded_images)} images")
            
            return {
                "answer": answer,
                "confidence": confidence,
                "sources_used": sources_used,
                "num_images": len(encoded_images)
            }
        
        except Exception as e:
            logger.error(f"Multimodal generation failed: {e}")
            return {"answer": f"Error: {e}", "confidence": "Low"}
    
    def _extract_confidence(self, answer: str) -> str:
        """Extract confidence level from generated answer."""
        answer_lower = answer.lower()
        if "confidence: high" in answer_lower or "high confidence" in answer_lower:
            return "High"
        elif "confidence: medium" in answer_lower or "medium confidence" in answer_lower:
            return "Medium"
        elif "confidence: low" in answer_lower or "low confidence" in answer_lower:
            return "Low"
        else:
            return "Medium"  # Default
    
    def _extract_cited_sources(self, answer: str) -> List[int]:
        """Extract source indices cited in the answer."""
        import re
        # Match [Source X] patterns
        matches = re.findall(r'\[Source (\d+)\]', answer)
        return [int(m) for m in matches]
    
    def _self_reflect(self, query: str, context: str, answer: str) -> Dict:
        """
        Perform self-reflection on generated answer.
        
        Uses a second LLM call to evaluate the quality of the answer.
        """
        reflection_prompt = SELF_REFLECTION_TEMPLATE.format(
            query=query,
            context=context[:2000],  # Truncate for token limits
            answer=answer
        )
        
        try:
            reflection = self.llm.invoke([
                {"role": "user", "content": reflection_prompt}
            ])
            
            # Parse reflection (simplified - could use structured output)
            reflection_text = reflection.content
            
            # Extract quality score
            import re
            score_match = re.search(r'quality score[:\s]+(\d+)', reflection_text.lower())
            quality_score = int(score_match.group(1)) if score_match else 5
            
            return {
                "reflection_text": reflection_text,
                "quality_score": quality_score
            }
        
        except Exception as e:
            logger.warning(f"Self-reflection failed: {e}")
            return {"reflection_text": "", "quality_score": 5}


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_answer(
    query: str,
    retrieval_results: Dict,
    model_name: str = "gpt-4o",
    enable_reflection: bool = False
) -> Dict:
    """
    End-to-end answer generation from retrieval results.
    
    Args:
        query: User question
        retrieval_results: Output from retrieval workflow (context, sources)
        model_name: LLM model to use
        enable_reflection: Enable quality checks
    
    Returns:
        Complete response with answer, sources, and confidence
    """
    generator = MultimodalGenerator(
        model_name=model_name,
        enable_reflection=enable_reflection
    )
    
    return generator.generate(
        query=query,
        context=retrieval_results["context"],
        sources=retrieval_results["sources"]
    )


# Example usage
if __name__ == "__main__":
    # Test with dummy data
    generator = MultimodalGenerator(model_name="gpt-3.5-turbo")  # Cheaper for testing
    
    sample_context = """[Source 1: sample.pdf]
The quick brown fox jumps over the lazy dog. This sentence contains all letters of the alphabet.

---
[Source 2: animals.txt]
Foxes are omnivorous mammals found across the Northern Hemisphere."""
    
    sample_sources = [
        {"index": 1, "source": "sample.pdf", "modality": "text"},
        {"index": 2, "source": "animals.txt", "modality": "text"}
    ]
    
    result = generator.generate(
        query="What is special about the sentence 'the quick brown fox'?",
        context=sample_context,
        sources=sample_sources
    )
    
    print("\n=== Generated Answer ===")
    print(result["answer"])
    print(f"\nConfidence: {result['confidence']}")
    print(f"Sources used: {result['sources_used']}")
    print(f"Generation time: {result['generation_time_ms']:.2f}ms")
