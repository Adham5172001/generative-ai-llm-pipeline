"""
Hallucination Evaluation for LLM Outputs
Author: Adham Aboulkheir
"""
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FaithfulnessScorer:
    """
    Score LLM response faithfulness to source context.
    Uses NLI-inspired approach with TF-IDF similarity as proxy.
    In production: use DeBERTa-NLI cross-encoder.
    """
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
    
    def score_sentence(self, sentence: str, context: str) -> float:
        """Score a single sentence against context."""
        if not sentence.strip() or not context.strip():
            return 0.0
        
        try:
            vec = TfidfVectorizer(stop_words="english")
            matrix = vec.fit_transform([sentence, context])
            sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
            return float(np.clip(sim * 2.5, 0, 1))
        except Exception:
            return 0.0
    
    def evaluate_response(self, response: str, context: str) -> Dict:
        """Evaluate full response for hallucination."""
        sentences = [s.strip() for s in response.split(".") if len(s.strip()) > 15]
        if not sentences:
            return {"faithfulness": 0.0, "hallucination_rate": 1.0, "n_sentences": 0}
        
        scores = [self.score_sentence(s, context) for s in sentences]
        hallucinated = [s for s in scores if s < self.threshold]
        
        return {
            "faithfulness": float(np.mean(scores)),
            "hallucination_rate": len(hallucinated) / len(scores),
            "n_sentences": len(sentences),
            "sentence_scores": scores,
            "hallucinated_count": len(hallucinated)
        }
    
    def batch_evaluate(self, responses: List[str], contexts: List[str]) -> List[Dict]:
        return [self.evaluate_response(r, c) for r, c in zip(responses, contexts)]


class BLEUScorer:
    """Compute BLEU score for text generation evaluation."""
    
    @staticmethod
    def ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
        result = {}
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i+n])
            result[gram] = result.get(gram, 0) + 1
        return result
    
    def score(self, hypothesis: str, reference: str, max_n: int = 4) -> float:
        hyp_tokens = hypothesis.lower().split()
        ref_tokens = reference.lower().split()
        
        if not hyp_tokens:
            return 0.0
        
        precisions = []
        for n in range(1, max_n + 1):
            hyp_ngrams = self.ngrams(hyp_tokens, n)
            ref_ngrams = self.ngrams(ref_tokens, n)
            
            matches = sum(min(count, ref_ngrams.get(gram, 0))
                          for gram, count in hyp_ngrams.items())
            total = sum(hyp_ngrams.values())
            precisions.append(matches / total if total > 0 else 0.0)
        
        if min(precisions) == 0:
            return 0.0
        
        log_avg = np.mean([np.log(p + 1e-10) for p in precisions])
        bp = min(1.0, np.exp(1 - len(ref_tokens) / (len(hyp_tokens) + 1e-9)))
        return float(bp * np.exp(log_avg))


if __name__ == "__main__":
    print("Hallucination Evaluation Demo")
    scorer = FaithfulnessScorer()
    bleu = BLEUScorer()
    
    context = "Error E-47 indicates a power supply fault. Check the 5A fuse and 12V power cable."
    
    test_cases = [
        ("The device has a power supply fault. Check the fuse.", "Faithful"),
        ("The device has a GPU failure. Replace the graphics card.", "Hallucinated"),
        ("Error E-47 means the 5A fuse needs replacement.", "Faithful"),
    ]
    
    print(f"\n{'Response':<55} {'Type':<12} {'Faithfulness':>12}")
    print("-" * 82)
    for response, label in test_cases:
        result = scorer.evaluate_response(response, context)
        bleu_score = bleu.score(response, context)
        print(f"  {response[:50]:<53} {label:<12} {result['faithfulness']:>12.1%}")
