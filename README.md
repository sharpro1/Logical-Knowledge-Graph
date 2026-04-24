# SymbolLKG: Logic Knowledge Graph Pipeline for Multi-hop QA

A neuro-symbolic framework that constructs knowledge graphs from text passages, retrieves relevant subgraphs via vector search and graph traversal, and augments LLM reasoning for multi-hop question answering and formal logic tasks.

## System Architecture

```
Input: Question Q + Passages P_1...P_n

Stage 1 — KG Construction (KGBuilder)
  Ontology schema (JSON) → LLM extraction prompt → chunked LLM calls → parse JSON
  → Build graph: nodes{} + edges[] + adjacency{}

Stage 2 — Subgraph Retrieval (KGRetriever)
  Question → BGE-M3 encode → vector search seed nodes → optional cross-encoder rerank
  → BFS 1-2 hop traversal → subgraph

Stage 3 — Answer Generation (LLM)
  Full passages + subgraph hints + question → LLM → chain-of-thought → Final Answer

Stage 4 — Evaluation
  Token-F1 + Exact Match (SQuAD standard) with alias support
```

## Key Features

- **Ontology-driven KG extraction**: Entity types (Person, Organization, Location, CreativeWork, Event, DateTime, Quantity) and 30+ relationship types defined in a JSON schema, dynamically injected into LLM prompts
- **Chunked extraction**: 20 paragraphs split into 5 batches of 4, each processed independently to avoid JSON output truncation (reduces empty-graph rate from 72% to <6%)
- **True graph structure**: Nodes + directed edges + bidirectional adjacency lists (not flat fact lists)
- **Two-stage retrieval**: BGE-M3 bi-encoder for candidate recall + optional BGE-reranker-v2-m3 cross-encoder for precision filtering + BFS graph traversal
- **"Direct superset" design**: KG mode passes all original passages plus structured KG hints, guaranteeing performance >= direct baseline
- **Multi-backend**: Memory-only (no database dependency) or Neo4j for production
- **Multi-LLM**: Supports SiliconFlow, Aliyun, OpenRouter (Llama, Claude, GPT, etc.)
- **Structured JSON logging**: Separate directories for LLM inputs, outputs, full KG, and retrieved subgraphs

## Project Structure

```
LKG_pipeline/
├── eval_musique.py              # Main evaluation script (multi-hop QA)
├── eval_end_to_end.py           # Router ablation (logic reasoning datasets)
├── run_pipeline.py              # Legacy pipeline with symbolic solvers
│
├── LKG_pipeline_1/              # Core package
│   ├── config.py                # Configuration (embedding model, backend selection)
│   ├── schemas/
│   │   └── multihop_qa_ontology.json   # KG ontology definition
│   └── core/
│       ├── kg_builder_v3.py     # KnowledgeGraph + KGBuilder + KGRetriever
│       ├── reranker.py          # Cross-encoder reranker (BGE-reranker-v2-m3)
│       ├── memory_graph.py      # Memory-based GraphManager (Neo4j replacement)
│       ├── memory_vector_store.py  # NumPy vector index + BM25
│       ├── llm_builder.py       # LLM-based KG extraction (legacy ontology)
│       ├── graph_ops.py         # Neo4j-based GraphManager
│       ├── router.py            # Logic solver router
│       ├── fact_extractor.py    # Flat fact extraction v1
│       └── fact_extractor_v2.py # Flat fact extraction v2
│
├── prompts/
│   ├── musique_qa.txt           # Multi-hop QA answer generation prompt
│   ├── z3_solver.txt            # Z3 code generation prompt
│   ├── prover9_solver.txt       # Prover9 code generation prompt
│   └── pyke_solver.txt          # Pyke code generation prompt
│
├── data/
│   ├── MuSiQue/                 # MuSiQue dataset + loader + metrics
│   ├── 2WikiMultiHopQA/         # 2WikiMultiHopQA dataset
│   ├── HotpotQA/                # HotpotQA dataset
│   ├── AR-LSAT/                 # Logic reasoning datasets
│   ├── FOLIO/
│   ├── LogicalDeduction/
│   ├── ProntoQA/
│   ├── ProofWriter/
│   └── load_multihop.py         # 2Wiki/HotpotQA data loader
│
├── solvers/                     # Symbolic solvers (Z3, Prover9, Pyke)
├── outputs/                     # Evaluation results (JSON)
├── musique_logs/                # Structured logs per run
│
├── run_20pct_eval.sh            # 20% test set evaluation (3 datasets)
├── run_all_multihop_100.sh      # 100-sample evaluation (3 datasets x 4 modes)
├── run_musique_100_v7.sh        # MuSiQue-only evaluation
└── download_bge_m3.py           # Model download helper
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (for embedding/reranker models)

### Dependencies

```bash
pip install openai sentence-transformers numpy torch
```

Optional:
```bash
pip install neo4j          # Only if using Neo4j backend
pip install modelscope     # For downloading models from Chinese mirrors
```

### Model Downloads

The system uses three models (downloaded automatically on first use):

| Model | Size | Purpose |
|-------|------|---------|
| `BAAI/bge-m3` | 2.3 GB | Embedding (1024D vectors for node retrieval) |
| `BAAI/bge-reranker-v2-m3` | 1.1 GB | Cross-encoder reranker (optional) |
| LLM (API) | — | KG extraction + answer generation |

To pre-download embedding models:

```bash
# Via ModelScope mirror (recommended for China mainland)
python download_bge_m3.py --modelscope

# Via HuggingFace mirror
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download BAAI/bge-m3
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download BAAI/bge-reranker-v2-m3
```

## Usage

### Quick Start

```bash
# Smoke test (5 samples, direct mode, no KG)
python eval_musique.py --dataset musique --n 5 --mode direct --num_workers 4

# With KG (recommended mode)
python eval_musique.py --dataset musique --n 100 --mode kg_v3 \
    --embedding bge-m3 --num_workers 4

# With KG + reranker
python eval_musique.py --dataset musique --n 100 --mode kg_v3 \
    --embedding bge-m3 --use_rerank --num_workers 4
```

### Evaluation Modes

| Mode | Input to LLM | KG Built? | Description |
|------|-------------|-----------|-------------|
| `direct` | All paragraphs | No | Baseline: LLM reads all passages |
| `oracle` | Gold paragraphs only | No | Upper bound (MuSiQue only) |
| `kg_v3` | All paragraphs + KG subgraph | Yes | **Recommended**: KG-augmented reasoning |
| `kg_v3_only` | KG subgraph only | Yes | Pure KG (no raw passages) |

### Switching LLM Providers

```bash
# SiliconFlow (default)
python eval_musique.py --n 100 --mode kg_v3

# OpenRouter + Llama 3.3 70B
python eval_musique.py --n 100 --mode kg_v3 \
    --llm openrouter:meta-llama/llama-3.3-70b-instruct

# OpenRouter + Claude
python eval_musique.py --n 100 --mode kg_v3 \
    --llm openrouter:anthropic/claude-3.5-sonnet
```

### Datasets

```bash
# MuSiQue (2-4 hop, 20 paragraphs per question)
python eval_musique.py --dataset musique --n 100 --mode kg_v3

# 2WikiMultiHopQA (2 hop, 10 paragraphs)
python eval_musique.py --dataset 2wiki --n 100 --mode kg_v3

# HotpotQA (2 hop, 10 paragraphs)
python eval_musique.py --dataset hotpotqa --n 100 --mode kg_v3
```

### Batch Evaluation Scripts

```bash
# 20% test set across all 3 datasets (MuSiQue=483, 2Wiki=200, HotpotQA=200)
bash run_20pct_eval.sh

# 100-sample quick evaluation
bash run_all_multihop_100.sh

# Filter by dataset or mode
bash run_all_multihop_100.sh musique         # Only MuSiQue
bash run_all_multihop_100.sh all direct      # Only direct mode
```

### Logic Reasoning (Solver-based)

For formal logic datasets (FOLIO, AR-LSAT, LogicalDeduction, ProntoQA, ProofWriter):

```bash
# Router ablation: LLM-based router vs random solver selection
python eval_end_to_end.py --samples_per_dataset 20 --backend memory --num_workers 4
```

## KG Construction Pipeline

### Step 1: Ontology-Driven Extraction

The system loads entity types and relationship types from `LKG_pipeline_1/schemas/multihop_qa_ontology.json`:

- **7 entity types**: Person, Organization, Location, CreativeWork, Event, DateTime, Quantity
- **30+ relationship types**: born_in, directed_by, founded_by, located_in, etc.
- **6 extraction rules**: evidence sourcing, IS_A filtering, cross-paragraph bridge prioritization

These are dynamically injected into the LLM prompt, constraining extraction quality.

### Step 2: Chunked LLM Extraction

20 paragraphs are split into 5 batches of 4. Each batch is independently processed by the LLM, producing structured JSON with entities and relationships. This reduces output truncation risk from 72% to <6%.

### Step 3: Graph Construction

Extracted entities and relationships are stored in a true graph structure:
- **Nodes**: `Dict[name_lower → {name, type, evidence, paragraph}]`
- **Edges**: `List[{source, target, predicate, evidence, paragraph}]`
- **Adjacency**: Bidirectional `Dict[name_lower → List[(edge_idx, neighbor)]]`

Same-name entities (case-insensitive) are automatically merged, creating cross-paragraph connections.

### Step 4: Vector Index + BFS Retrieval

1. All node names are encoded with BGE-M3 (1024D)
2. Question is encoded and matched against nodes via cosine similarity
3. Top-8 seed nodes are selected (threshold > 0.35)
4. Optional: cross-encoder reranker re-scores top-30 candidates
5. BFS traverses 1-2 hops from seeds along the adjacency list
6. Retrieved subgraph (avg 24 nodes, 22 edges) is formatted as text

## Output Structure

### Evaluation Results

```
outputs/
  eval_musique_kg_v3_bge-m3_rerank_20260423_xxx.json
  eval_musique_2wiki_direct_20260423_xxx.json
  ...
```

Each JSON contains `config`, `metrics` (overall EM/F1, per-hop breakdown), and `details` (per-sample predictions).

### Structured Logs

```
musique_logs/<mode>_<embedding>_<rerank>_<timestamp>/
  input/
    <sample_id>_llm_input.json     # Full LLM prompt
  output/
    <sample_id>_llm_output.json    # LLM response + prediction + EM/F1
  kg/
    <sample_id>_full_kg.json       # Complete extracted KG (all nodes + edges)
  subgraph/
    <sample_id>_subgraph.json      # Retrieved subgraph (seeds + BFS result)
```

## Experimental Results

### MuSiQue (n=100, DeepSeek-V3.2, BGE-M3)

| Mode | EM | F1 | Description |
|------|----|----|-------------|
| direct | 54.0% | 63.4% | Baseline |
| **kg_v3** | **58.0%** | **69.4%** | +6.0 F1 over baseline |


## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_BACKEND` | `minilm` | Embedding model: `minilm`, `bge-base`, `bge-large`, `bge-m3` |
| `LKG_BACKEND` | `neo4j` | Graph backend: `memory` (recommended), `neo4j` |

### API Keys

Configure in `eval_musique.py` or via `--llm` parameter:

| Provider | Format |
|----------|--------|
| SiliconFlow | `--llm siliconflow:Pro/deepseek-ai/DeepSeek-V3.2` |
| Aliyun | `--llm aliyun:deepseek-v3.2` |
| OpenRouter | `--llm openrouter:meta-llama/llama-3.3-70b-instruct` |

## Citation

If you use this system in your research, please cite:

```bibtex
@article{symbollkg2025,
  title={SymbolLKG: Logic Knowledge Graph for Neuro-Symbolic Reasoning},
  year={2025}
}
```

## License

This project is for academic research purposes.
