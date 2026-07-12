# Generative AI Pipeline Prototype

An offline educational demonstration of LLM-pipeline configuration, small-document retrieval, and illustrative evaluation.

## Implemented

- LoRA configuration and parameter-count utilities
- Small deterministic retrieval example
- Token-overlap faithfulness heuristic
- Illustrative benchmark reporting

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py
```

The demo does not fine-tune or call a real language model. Its benchmark values and hallucination scores are illustrative and must not be interpreted as measured model performance.

## License

[MIT](LICENSE)
