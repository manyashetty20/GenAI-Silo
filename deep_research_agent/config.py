"""Configuration for the agentic deep research system."""
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# deep_research_agent/config.py

@dataclass
class LLMConfig:
    # 1. Change backend to "openai"
    backend: str = "openai" 
    
    # 2. Use Llama 3.3 70B for journal-quality reasoning
    model: str = "llama-3.3-70b-versatile" 
    
    # 3. Set the Groq Base URL
    base_url: str = "https://api.groq.com/openai/v1"
    
    # 4. Point to the API key in your .env
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    
    # 5. Optional: Keep temperature low for academic synthesis
    temperature: float = 0.1
    max_tokens: int = 4096

@dataclass
class EmbeddingConfig:
    """Embedding model for retrieval (CPU-friendly)."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Alternative for scientific text: "allenai/scibert_scivocab_uncased"
    device: str = "cpu"


# deep_research_agent/config.py

@dataclass
class RetrievalConfig:
    """Retrieval limits and sources."""
    max_arxiv_results: int = 15
    max_semantic_scholar_results: int = 10
    top_k_after_rerank: int = 10
    # Add this line to fix the AttributeError
    end_date: Optional[str] = None  # Format: YYYY-MM-DD


@dataclass
class AgentConfig:
    """Agent loop and verification."""
    max_plan_iterations: int = 3
    max_verify_iterations: int = 2
    min_citations_per_claim: int = 1


@dataclass
class Config:
    """Full pipeline config."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


def get_config() -> Config:
    return Config()
