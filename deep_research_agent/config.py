"""Configuration for the agentic deep research system."""
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class LLMConfig:
    """LLM endpoint and model settings.

    backend:
      - "openai": use OpenAI-compatible API via langchain_openai.ChatOpenAI
      - "hf_local": use a local Hugging Face model (optionally with LoRA)
    """

    backend = "hf_local"
    model: str = "gpt-4o-mini"  # for backend=="openai", this is the OpenAI model name
    temperature: float = 0.3
    max_tokens: int = 8192
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))  # e.g. http://localhost:11434/v1 for Ollama

    # For backend == "hf_local"
    hf_model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hf_lora_path = "lora_writer_tinyllama"

@dataclass
class EmbeddingConfig:
    """Embedding model for retrieval (CPU-friendly)."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Alternative for scientific text: "allenai/scibert_scivocab_uncased"
    device: str = "cpu"


@dataclass
class RetrievalConfig:
    """Retrieval limits and sources."""
    max_arxiv_results: int = 30
    max_semantic_scholar_results: int = 20
    top_k_after_rerank: int = 15
    end_date: Optional[str] = None  # YYYY-MM-DD for reproducibility


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
