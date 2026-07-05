# Energy Sector Agentic vs RAG: A Comparative Benchmark Study

This repository presents a comprehensive benchmark comparing **traditional retrieval-augmented generation (RAG)** and **agentic multi-tool pipelines** for table-grounded decision support in energy-sector sustainability reporting.

---

##  Study Overview

### Pipeline Architectures

#### Traditional RAG Pipeline
A linear, deterministic pipeline optimized for simplicity and baseline performance.

**Architecture Flow:**
```
Question Instance 
    ↓
[Grounded Retrieval Enabled?] → No → INSUFFICIENT_CONTEXT
    ↓ Yes
Dense Vector Retrieval (k=3 with structured constraints)
    ↓
Semantic Reranking
    ↓
[LLM Synthesis Available?] → No → [Unregistered Record Has Value?]
    ↓ Yes                                      ↓ Yes
LLM-grounded Answer Generation      Extract & Normalize Primary Value
    ↓                                          ↓
    └──→ Construct Citation Metadata ←─────────┘
         ↓
    Final Prediction (Answer + Citations + Trace)
```

**Key Metrics:**
- **Exact Match**: 34.2%
- **Latency**: 10.8s
- **Retrieval Time**: 71% of total
- **Faithfulness**: 50.6%

---

#### Agentic Multi-Tool Pipeline
An adaptive pipeline with strategy-driven routing and dynamic expansion.

**Architecture Flow:**
```
Question Instance
    ↓
Strategy Classification (5 types)
    ↓
Corpus Retrieval (k=3/10/25, strategy-dependent)
    ↓
Semantic Reranking
    ↓
Candidate Sort
    ↓
[Evidence Sufficient?] → No → Retry: Expand Query ──┐
    ↓ Yes                                            │
[Multi-table Expand?] → Yes → Expand: Multi-source ─┤
    ↓ No                                             │
    └─────────────────────────────────────────────┬──┘
                                                  ↓
        ╔═══════════════════════════════════════════╗
        ║   6-PRIORITY ANSWER CASCADE              ║
        ║  ┌─────────────┬─────────────┬────────┐ ║
        ║  │ P0M: GPT    │ P0A: Direct │ P0B:   │ ║
        ║  │ Multi-table │ Cell Lookup │ Schema │ ║
        ║  └─────────────┴─────────────┴────────┘ ║
        ║  ┌─────────────┬─────────────┬────────┐ ║
        ║  │ P0C: Multi- │ P1: Calc    │ P2:    │ ║
        ║  │ join        │ (≥0.85 conf)│Default │ ║
        ║  └─────────────┴─────────────┴────────┘ ║
        ║            ↓ (Merge flows)              ║
        ║        Answer Selected                   ║
        ║        (Highest Priority)                ║
        ╚═══════════════════════════════════════════╝
         ↓
    Answer Selected & Verified
    ↓
    Final Prediction
```

**Key Metrics:**
- **Exact Match**: 53.4% (+56% vs RAG)
- **Latency**: 5.0s (-54% vs RAG)
- **Faithfulness**: 65.0% (+28% vs RAG)

---

## Research Questions

1. **RQ1: Effectiveness**  
   Do agentic multi-tool pipelines improve faithfulness and transparency compared to traditional RAG on table-grounded sustainability questions?

2. **RQ2: Trade-offs**  
   How do correctness, citation quality, and latency trade off between the two systems under identical benchmarks and evaluation protocols?

3. **RQ3: Error Patterns**  
   Which error types dominate each pipeline? (unsupported claims, miscitations, quantitative errors, reasoning failures)

**See detailed operationalization in [`docs/research_questions.md`](docs/research_questions.md).**

---

## � Dataset Source

This benchmark is built on the **GRI-QA dataset**, a comprehensive question-answering dataset for sustainability reporting:

**GRI-QA Benchmark Data Repository**: [https://github.com/softlab-unimore/gri_qa](https://github.com/softlab-unimore/gri_qa)

### Dataset Characteristics
- **Domain**: Global Reporting Initiative (GRI) Sustainability Reports
- **Focus**: Energy sector sustainability disclosure
- **Format**: Structured Q&A pairs with table grounding
- **Size**: 1000+ complex questions with multi-table context
- **Benchmark Splits**: Selected subsets used for standardized evaluation

### Using the Dataset
To use this benchmark, clone or fork the original GRI-QA repository and use the `prepare_data.py` script to import selected questions:

```bash
# Clone GRI-QA repository
git clone https://github.com/softlab-unimore/gri_qa.git ../gri_qa

# Prepare benchmark data from GRI-QA
python scripts/prepare_data.py --source-root ../gri_qa/dataset --target-root data/benchmark
```

---

##  Repository Structure

```
Energy_Sector_Agentic_vs_RAG/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project metadata
├── .env.example                       # Configuration template
├── .gitignore                         # Git ignore rules
├── Makefile                           # Build automation
│
├── src/                               # Core implementation
│   ├── gri_benchmark/                 # Benchmark framework
│   │   ├── pipelines/                 # RAG & Agentic implementations
│   │   ├── retrieval/                 # Retrieval components
│   │   ├── scoring/                   # Evaluation metrics
│   │   └── utils/                     # Utility functions
│   └── agents/                        # Agent implementations
│
├── tests/                             # Unit & integration tests
│   ├── test_scoring.py
│   ├── test_retrieval.py
│   ├── test_pipelines.py
│   └── test_semantic_integration.py
│
├── configs/                           # Benchmark configurations
│   ├── benchmark.yaml                 # Default configuration
│   ├── benchmark_full.yaml            # Full dataset
│   ├── benchmark_small_batch_50.yaml  # Quick test (50 samples)
│   └── benchmark_ablation_*.yaml      # Ablation study configs
│
├── data/                              # Datasets & benchmarks
│   ├── benchmark/                     # Selected benchmark data (CSV)
│   ├── corpus/                        # Built retrieval corpus
│   └── dataset/                       # Raw GRI-QA data reference
│
├── docs/                              # Documentation
│   ├── research_questions.md          # RQ operationalization
│   ├── evaluation_protocol.md         # Evaluation methodology
│   ├── annotation_guide.md            # Manual annotation guidelines
│   ├── complex_benchmark_guide.md     # Complex subset details
│   └── *.md                           # Additional analysis
│
├── scripts/                           # Utility & benchmark scripts
│   ├── prepare_data.py                # Prepare benchmark data
│   ├── build_retrieval_corpus.py      # Build retrieval indexes
│   ├── run_benchmark.py               # Execute benchmark
│   └── run_ablation_compare.py        # Ablation studies
│
├── results/                           # Benchmark outputs (git-ignored)
│   ├── RAG_predictions.json           # RAG pipeline predictions
│   ├── Agentic_predictions.json       # Agentic pipeline predictions
│   ├── summary.json                   # Aggregate metrics
│   ├── ablation/                      # Ablation study results
│   └── analysis/                      # Analysis reports
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git
- ~2GB disk space

### Setup Instructions

```bash
# 1. Clone repository
git clone <repository-url>
cd Energy_Sector_Agentic_vs_RAG

# 2. Create Python environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (if using APIs)
cp .env.example .env
# Edit .env with your settings

# 5. Prepare data
python scripts/prepare_data.py --source-root ../dataset --target-root data/benchmark

# 6. Build retrieval corpus
PYTHONPATH=src python scripts/build_retrieval_corpus.py --config configs/benchmark.yaml

# 7. Run benchmark
PYTHONPATH=src python scripts/run_benchmark.py --config configs/benchmark.yaml

# Results are written to results/
```

### Using Makefile (Recommended)

```bash
make setup           # Setup environment & install dependencies
make prepare-data    # Prepare benchmark datasets
make build-corpus    # Build retrieval corpus
make benchmark       # Run full benchmark
make benchmark-quick # Run quick test (50 samples)
make ablation        # Run ablation studies
make clean           # Remove generated artifacts
help                 # Show all available targets
```

---

##  Understanding Results

### Output Files

Benchmark results are generated in `results/`:

- `RAG_predictions.json` - Traditional RAG outputs (question, answer, citations, metadata)
- `Agentic_predictions.json` - Agentic pipeline outputs
- `summary.json` - Aggregate metrics (EM, latency, faithfulness, etc.)
- `comparison_report.md` - Detailed comparative analysis

### Key Metrics

| Metric | Description | RAG | Agentic | Delta |
|--------|-------------|-----|---------|-------|
| **Exact Match (%)** | Predicted answer matches ground truth | 34.2 | 53.4 | +56% ✓ |
| **Latency (s)** | End-to-end processing time | 10.8 | 5.0 | -54% ✓ |
| **Faithfulness (%)** | Citations support predictions | 50.6 | 65.0 | +28% ✓ |
| **Retrieval Time (%)** | Retrieval as % of total | 71 | - | - |

### Error Taxonomy

Errors are classified as:
- **Unsupported Claims**: Predicted value not in retrieved context
- **Miscitations**: Wrong table/cell cited for correct value
- **Quantitative Errors**: Arithmetic or aggregation mistakes
- **Reasoning Failures**: Incorrect tool sequencing or strategy selection

---

##  Configuration

### Benchmark Config (`configs/benchmark.yaml`)

```yaml
# Dataset selection
dataset:
  source: ../dataset
  target: data/benchmark
  subset: selected_1000

# Retrieval parameters
retrieval:
  top_k: [3, 10, 25]
  similarity_metric: cosine
  reranking: semantic
  corpus_type: strict  # or lenient

# LLM configuration
llm:
  model: gpt-4
  temperature: 0.0
  max_tokens: 500
  timeout: 30

# Evaluation
evaluation:
  metrics: [exact_match, faithfulness, latency, citation_quality]
  annotation_scheme: strict

# Agentic-specific
agentic:
  strategy_classification: enabled
  dynamic_expansion: enabled
  cascade_priorities: 6
```

### Environment Variables (`.env`)

```bash
# Azure/OpenAI API
PROJECT_ENDPOINT=https://your-endpoint.com
API_KEY=your-api-key
MODEL_DEPLOYMENT=deployment-name
EMBEDDING_DEPLOYMENT=embedding-deployment

# Optional
DEBUG=false
LOG_LEVEL=INFO
RANDOM_SEED=42
```

##  Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_scoring.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run integration tests
pytest tests/test_semantic_integration.py -v
```

---

## Experiment Reproducibility

### Controlling Randomness

```bash
# Set random seed for reproducible results
export RANDOM_SEED=42
PYTHONPATH=src python scripts/run_benchmark.py \
  --config configs/benchmark.yaml \
  --seed 42
```

### Benchmark Variants

- **Small Test**: `benchmark_small_batch_50.yaml` (50 samples, ~2 min)
- **Standard**: `benchmark.yaml` (1000 samples, ~30 min)
- **Full**: `benchmark_full.yaml` (all data, ~2 hours)
- **Ablation Studies**: `benchmark_ablation_*.yaml` (component analysis)

### Ablation Workflow

```bash
# Run direct strict vs non-strict comparison
PYTHONPATH=src python scripts/run_ablation_compare.py --rebuild-corpus

# Results: results/ablation/ablation_compare.json
```

---

##  Core Modules

### `src/gri_benchmark/`

**Pipelines**
- `rag_baseline.py` - Traditional RAG implementation
- `agentic_pipeline.py` - Agentic multi-tool implementation

**Retrieval**
- `dense_retriever.py` - Dense vector retrieval (BM25 + embeddings)
- `reranker.py` - Semantic reranking
- `corpus_builder.py` - Index construction

**Scoring**
- `exact_match.py` - EM metric
- `faithfulness.py` - Citation-based faithfulness
- `latency.py` - End-to-end timing
- `error_taxonomy.py` - Error classification

**Utilities**
- `data_loader.py` - Load benchmark data
- `metrics.py` - Aggregate metric computation
- `logging.py` - Unified logging

---

##  Dependencies

See [`requirements.txt`](requirements.txt) for complete list.

**Core:**
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - ML utilities

**LLM & Retrieval:**
- `openai` - GPT models
- `azure-*` - Azure services
- `sentence-transformers` - Dense embeddings

**Evaluation:**
- `pytest` - Testing framework
- `rouge-score` - Text similarity

**Visualization:**
- `matplotlib` - Diagrams & plots

---

## 🛠️ Troubleshooting

### Common Issues

**`ModuleNotFoundError: gri_benchmark`**
```bash
export PYTHONPATH=src
```

**Data preparation fails**
```bash
# Verify source path
ls -la ../dataset
# Check CSV format has required columns
```

**API authentication errors**
```bash
# Verify .env configuration
cat .env
# Test connection
python scripts/sanity_live_check.py
```

**Insufficient memory**
```bash
# Use smaller batch size
--config configs/benchmark_small_batch_50.yaml
```

---

##  Citation

If you use this benchmark in your research, please cite both this work and the original GRI-QA dataset:

### Benchmark Study
```bibtex
@dataset{agentic_vs_rag_2024,
  title={Energy Sector Agentic vs RAG: A Comparative Benchmark Study},
  author={Jyoti Panda},
  year={2026},
  institution={Liverpool John Moores University},
  note={Available at: https://github.com/Energy-Sector-Agentic-vs-RAG-A-Comparative-Benchmark-Study}
}
```

### Original GRI-QA Dataset
```bibtex
@dataset{gri_qa,
  title={GRI-QA: A Question Answering Dataset for Sustainability Reporting},
  author={University of Modena and Reggio Emilia - UNIMORE},
  url={https://github.com/softlab-unimore/gri_qa},
  note={Original dataset source for this benchmark}
}
```

---

## Contact & Support

For questions or issues:

1. **Check documentation** in `docs/`
2. **Review examples** in `scripts/`
3. **Run tests** to verify setup: `pytest tests/`
4. **Create an issue** with:
   - Description and steps to reproduce
   - Expected vs actual behavior
   - Environment (OS, Python version, etc.)

---

## Security & Privacy

- ✅ All code is open-source
- ✅ No credentials in repository
- ✅ Use `.env.example` template only
- ✅ Keep actual `.env` file untracked
- ✅ Safe for peer review
- ✅ Publish results, not private data

---

##  Additional Resources

- [Research Questions](docs/research_questions.md) - Detailed RQ operationalization
- [Evaluation Protocol](docs/evaluation_protocol.md) - Methodology & metrics
- [Annotation Guide](docs/annotation_guide.md) - Manual annotation process
- [Findings & Observations](docs/findings_observations.md) - Running analysis log

---

**Version**: 1.0  
**Last Updated**: 2026-07-06  
**Status**: Publication Ready ✓
