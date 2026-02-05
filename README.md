# RAGTurk: Best Practices for Retrieval-Augmented Generation in Turkish

[![arXiv](https://img.shields.io/badge/arXiv-2602.03652-b31b1b.svg)](https://arxiv.org/abs/2602.03652)

## Overview

**RAGTurk** investigates best practices for **Retrieval-Augmented Generation (RAG)** systems in **Turkish**, a morphologically rich and low-resource language.  
While RAG has become a standard technique for improving factuality and grounding in large language models, existing benchmarks and design guidelines are overwhelmingly **English-centric**.

This repository documents the **arXiv version** of the paper, which has been **accepted to EACL 2026 (SIGTURK)**.  
The final camera-ready version will be published in the official conference proceedings.

Dataset available on Hugging Face: [Hugging Face Dataset Link](PUT_HF_DATASET_LINK_HERE)

Paper (arXiv): https://arxiv.org/abs/2602.03652  
Conference: **EACL 2026 – SIGTURK**

---

## Motivation

- RAG pipelines are widely adopted to reduce hallucinations and improve factual accuracy.
- Most RAG design choices are validated only on English data.
- Turkish introduces challenges due to:
  - Agglutinative morphology
  - Tokenization mismatch
  - Query–document lexical divergence

**RAGTurk** aims to identify which RAG components matter most for Turkish and which introduce unnecessary complexity.

---

## Contributions

1. **Turkish RAG Benchmark Dataset**
   - Constructed from Turkish Wikipedia and CulturaX
   - Question–answer pairs aligned with retrieved passage chunks
   - Enables reproducible evaluation of Turkish RAG systems

2. **End-to-End RAG Pipeline Analysis**
   - Systematic evaluation of the full RAG pipeline
   - No task-specific supervised fine-tuning

3. **Best Practice Recommendations**
   - Identifies high-performing and cost-efficient configurations
   - Demonstrates that over-stacking generative modules degrades performance

4. **Language-Specific Insights**
   - Shows how Turkish morphology affects retrieval and generation
   - Highlights the limits of English-centric RAG heuristics

---

## Evaluated RAG Pipeline Components

- Query Transformation
- Dense Retrieval
- Reranking (bi-encoder vs cross-encoder)
- Context Augmentation
- Answer Fusion
- Answer Refinement
- Post-processing

All components are evaluated independently and in combination.

---

## Key Findings

- **HyDE (Hypothetical Document Embeddings)** achieves the highest accuracy (~85%) at higher cost.
- **Cross-encoder reranking + context augmentation** provides a near-optimal trade-off (~84.6%).
- Excessive generative refinement harms performance in Turkish.
- Retrieval and reranking dominate overall RAG quality.

---

## What Makes RAGTurk Different

| Aspect | Prior RAG Work | RAGTurk |
|------|---------------|---------|
| Language Focus | English | Turkish |
| Pipeline Coverage | Partial | Full |
| Cost Analysis | Rare | Explicit |
| Morphology Awareness | Limited | Core |

---

## Use Cases

- Turkish QA and assistant systems
- Multilingual RAG benchmarking
- Low-resource language RAG research
- Cost-aware RAG system design

---

## Limitations

- Focused exclusively on Turkish
- Open-domain data only
- No supervised fine-tuning

---

## Conclusion

**RAGTurk** provides the first systematic evaluation of RAG pipeline design choices for Turkish and has been **accepted to EACL 2026 SIGTURK**.  
The results demonstrate that effective RAG systems must be adapted to linguistic structure, not only model capability.

---

## Citation

```bibtex
@inproceedings{ragturk2026,
  title={RAGTurk: Best Practices for Retrieval-Augmented Generation in Turkish},
  author={Kose, Suha Kagan and Baytekin, Mehmet Can and Aktas, Burak and Gorur, Bilge Kaan and Munis, Evren Ayberk and Yilmaz, Deniz and Kartal, Muhammed Yusuf and Toraman, Cagri},
  booktitle={Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  note={Accepted to SIGTURK. arXiv:2602.03652},
  year={2026}
}

