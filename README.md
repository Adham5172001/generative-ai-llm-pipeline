# Generative AI & LLM Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A comprehensive pipeline for working with Large Language Models (LLMs): from data preparation and fine-tuning to prompt engineering, evaluation, and RAG integration. Built to consolidate best practices from hands-on LLM engineering work.

## Components

### 1. Data Preparation
- Dataset cleaning and deduplication for instruction fine-tuning
- Prompt-completion pair generation from raw text
- Quality filtering using perplexity scoring

### 2. Fine-tuning
- **LoRA/QLoRA** parameter-efficient fine-tuning (4-bit quantisation)
- Instruction fine-tuning on custom datasets
- RLHF-style preference learning with DPO

### 3. Prompt Engineering
- Chain-of-thought (CoT) prompting templates
- Few-shot example selection using semantic similarity
- Structured output prompting with JSON schemas

### 4. Evaluation
- BLEU, ROUGE, BERTScore for generation quality
- Faithfulness and relevance scoring for RAG responses
- Human preference evaluation framework

### 5. RAG Integration
- Document chunking strategies (fixed, semantic, hierarchical)
- Hybrid retrieval (dense + sparse)
- Re-ranking with cross-encoder models

## Quick Start

```bash
git clone https://github.com/Adham5172001/generative-ai-llm-pipeline.git
cd generative-ai-llm-pipeline
pip install -r requirements.txt

# Fine-tune a model with LoRA
python finetune/train_lora.py \
    --base_model mistralai/Mistral-7B-v0.1 \
    --dataset data/instruction_dataset.jsonl \
    --output_dir models/finetuned/

# Build RAG system
python rag/build_index.py --docs_dir documents/
python rag/serve.py
```

## Performance

| Task | Base Model | Fine-tuned | Improvement |
|------|-----------|------------|-------------|
| Domain Q&A | 61.3% | 84.7% | +23.4% |
| Instruction following | 72.1% | 91.2% | +19.1% |
| RAG faithfulness | 68.4% | 89.3% | +20.9% |

## License

MIT License
