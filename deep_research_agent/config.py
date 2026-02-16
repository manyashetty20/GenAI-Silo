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
    backend: str = "openai" 
    
    # Switch to Qwen 2.5 72B - it's excellent for research and highly available
    model: str = "Qwen/Qwen2.5-72B-Instruct" 
    
    base_url: str = "https://router.huggingface.co/v1"
    
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("HF_TOKEN")
    )
    
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
