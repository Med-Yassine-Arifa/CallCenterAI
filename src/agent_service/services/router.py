"""
Routeur intelligent pour choisir le bon modèle
"""
import logging
from typing import Tuple

import langdetect
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

MAX_LENGTH = 40  # words
COMPLEXITY_THRESHOLD = 0.5  # probability threshold (0=easy, 1=very complex)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Load lightweight reasoning model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(DEVICE)


class ModelRouter:
    """Classe pour router vers le bon modèle"""

    def __init__(
        self,
        tfidf_url: str = "http://localhost:8001",
        transformer_url: str = "http://localhost:8002",
    ):
        self.tfidf_url = tfidf_url
        self.transformer_url = transformer_url

    # --------------------------
    # COMPLEXITY SCORING FUNCTION
    # --------------------------
    def estimate_complexity(self, text: str) -> float:
        """
        Uses Flan-T5 to score complexity as a probability between 0 and 1.
        """

        prompt = (
            "Evaluate the complexity of the following support ticket. "
            "Consider factors such as the number of concepts, technical difficulty, "
            "ambiguity, and the effort required to understand and resolve it. "
            "Return a single number between 0.0 and 1.0, where 0.0 means very simple "
            "and easy to understand, and 1.0 means highly complex and difficult. "
            "Respond with only the numeric value.\n\n"
            f"Text: {text}"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, max_new_tokens=10)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Parse numeric output safely
        try:
            score = float(result)
            score = max(0.0, min(score, 1.0))  # clamp to [0,1]
        except ValueError:
            score = 0.5  # fallback

        return score

    # --------------------------
    # ROUTING FUNCTION
    # --------------------------

    def choose_model(self, text: str) -> Tuple[str, str, dict]:
        # 1. Language detection
        try:
            lang = langdetect.detect(text)
        except Exception:
            lang = "unknown"

        if lang != "en":
            model = "distilbert"
        else:
            # 2. Length check
            word_count = len(text.split())
            if word_count > MAX_LENGTH:
                model = "distilbert"
            else:
                # 3. Complexity check
                complexity = self.estimate_complexity(text)
                model = "distilbert" if complexity > COMPLEXITY_THRESHOLD else "tfidf"

        url = self.tfidf_url if model == "tfidf" else self.transformer_url

        explanation = {
            "language": lang,
            "model_chosen": model,
            "url": f"{url}/predict",
        }

        return model, url, explanation
