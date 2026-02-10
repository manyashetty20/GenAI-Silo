# Agentic Deep Research System (DeepScholar-Optimized)

An **agentic deep research system** that outperforms the linear DeepScholar baseline by using a **Plan → Execute → Verify** loop, specialized agents, and optional LoRA adaptation.

## Features

- **Planner Agent**: Decomposes the benchmark query into sub-questions and search queries.
- **Search Agent**: Recursive literature search over **arXiv** and **Semantic Scholar** with embedding rerank.
- **Reader / Nugget Agent**: Extracts structured facts and claims from papers (nuggets) instead of full summaries.
- **Synthesizer / Writer Agent**: Produces a structured, cited Markdown report (DeepScholar-style).
- **Verifier / Skeptic Agent**: Audits claim–citation pairs and corrects the report.

Output is **Markdown with inline citations and references**, suitable for the [DeepScholar-bench](https://github.com/guestrin-lab/deepscholar-bench) evaluation suite.

## Setup
```bash
# From repo root (GenAI)
cd /path/to/project
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r deep_research_agent/requirements.txt
```

### Environment (OpenAI or local HF)

Create a `.env` in `GenAI` or `deep_research_agent`:
- **Groq backend** (Optimized for Agentic loops):
  - `GROQ_API_KEY=gsk_...`
- **OpenAI backend** (`backend="openai"` in `LLMConfig`):
  - `OPENAI_API_KEY=sk-...`
  - optional: `OPENAI_BASE_URL` for OpenAI-compatible endpoints (e.g. Ollama).
- **Local HF backend** (`backend="hf_local"` in `LLMConfig`):
  - set `hf_model_name_or_path` (e.g. local LLaMA/Mistral checkpoint)
  - optional: `hf_lora_path` pointing to LoRA adapter weights.

Embeddings use **sentence-transformers** (CPU-friendly); first run will download the model.

## Usage

### Single query (report to stdout)

```bash
# From GenAI directory
python -m deep_research_agent.run --query "What are the latest developments in retrieval-augmented generation?"
```

### With end date (for benchmark reproducibility)

```bash
python -m deep_research_agent.run --query "Your question" --end-date 2025-01-01
```

### Save report to file

```bash
python -m deep_research_agent.run --query "Your question" --output report.md
```

### Output for DeepScholar evaluation

DeepScholar-bench expects results in a folder (e.g. one file per query). Use:

```bash
python -m deep_research_agent.run --query "Your question" --output-dir results/agentic_system --query-id 0
```

Then run the official eval (from a clone of [deepscholar-bench](https://github.com/guestrin-lab/deepscholar-bench)):

```bash
python -m eval.main \
  --modes your_system \
  --evals organization nugget_coverage reference_coverage cite_p \
  --input_folder results/agentic_system \
  --output_folder results \
  --dataset_path dataset/related_works_combined.csv \
  --model_name gpt-4o
```

## Project structure

```
deep_research_agent/
  config.py           # LLM, embedding, retrieval, agent config
  llm.py              # LLM client (OpenAI or local HF/LoRA)
  run.py              # CLI entry
  agents/
    planner.py        # Query → sub-questions + search queries
    search_agent.py   # arXiv + Semantic Scholar + rerank
    reader.py         # Nugget extraction
    synthesizer.py    # Cited Markdown report
    verifier.py       # Claim–citation audit
  retrieval/
    arxiv_client.py
    semantic_scholar.py
    embeddings.py
  graph/
    workflow.py       # Plan → Execute → Verify loop
  training/
    prepare_writer_dataset.py  # Build writer dataset (query + nuggets + refs -> text)
    train_lora_writer.py       # LoRA fine-tuning script for writer model
```

## Configuration

Edit `config.py` or use env:

- **LLM**: `OPENAI_API_KEY`, `OPENAI_BASE_URL` (e.g. Ollama), model name.
- **Embeddings**: `model_name` (e.g. `allenai/scibert_scivocab_uncased` for scientific text).
- **Retrieval**: `max_arxiv_results`, `top_k_after_rerank`, `end_date`.
- **Agent**: `max_verify_iterations`, `max_plan_iterations`.

## Training & LoRA fine-tuning (from project idea)

- **Dataset prep**: Use `training/prepare_writer_dataset.py` to convert a DeepScholar-style CSV
  (with query, nuggets, references, target) into a JSONL of `{input, output}` pairs.
- **LoRA training**: Run `training/train_lora_writer.py` to fine-tune a local HF model on this dataset.
- **Use the adapter**: Point `hf_model_name_or_path` and `hf_lora_path` in `LLMConfig` and set `backend="hf_local"` to run the agentic pipeline with your adapted writer model.
- **Benchmark**: Clone DeepScholar-bench, generate answers with this system, and run their evaluation scripts to compare against DeepScholar-base and other systems.

## Step-by-step: run everything (fresh machine)

1) Clone and set up
```bash
cd /path/to/GenAI
python -m venv .venv
source .venv/bin/activate
pip install -r deep_research_agent/requirements.txt
```

2) Choose LLM backend  
- **OpenAI**: set `export OPENAI_API_KEY=sk-...` (backend stays `openai`).  
- **Local HF**: in `config.py -> LLMConfig`, set:
  ```python
  backend = "hf_local"
  hf_model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  hf_lora_path = None  # fill after training
  ```

3) Get DeepScholar data and build training JSONL
```bash
git clone https://github.com/guestrin-lab/deepscholar-bench.git
python -m deep_research_agent.training.prepare_writer_dataset \
  --csv deepscholar-bench/dataset/related_works_combined.csv \
  --out writer_train.jsonl
```

4) Train the LoRA writer (downloads model weights on first run; can take time)
```bash
python -m deep_research_agent.training.train_lora_writer \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train_jsonl writer_train.jsonl \
  --output_dir lora_writer_tinyllama
```

5) Use the trained adapter
In `config.py -> LLMConfig`:
```python
backend = "hf_local"
hf_model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
hf_lora_path = "lora_writer_tinyllama"
```

6) Run the agentic pipeline
```bash
python -m deep_research_agent.run \
  --query "What are recent developments in RAG?" \
  --end-date 2025-01-01 \
  --output report.md
```

7) (Optional) Produce outputs for DeepScholar eval
```bash
python -m deep_research_agent.run \
  --query "Your question" \
  --output-dir results/agentic_system \
  --query-id 0
# then run eval from deepscholar-bench with that folder
```

## Why training/download can be slow
- First run downloads the base model weights (e.g., TinyLlama ≈2 GB) from Hugging Face.
- LoRA training does a full pass over the dataset; on CPU it will take noticeably longer. On GPU it’s much faster. You can reduce epochs or batch size to shorten time, at the cost of quality.
