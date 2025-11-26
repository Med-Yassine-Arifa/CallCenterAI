import langdetect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# --------------------------
# CONFIG
# --------------------------
MAX_LENGTH = 40        # words
COMPLEXITY_THRESHOLD = 0.5  # probability threshold (0=easy, 1=very complex)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load lightweight reasoning model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(DEVICE)


# --------------------------
# COMPLEXITY SCORING FUNCTION
# --------------------------
def estimate_complexity(text: str) -> float:
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
def route_ticket(text: str) -> str:
    """
    Returns: 'svm' or 'distilbert'
    Decision logic:
        - If non-English → DistilBERT
        - If long and complex → DistilBERT
        - If long → DistilBERT
        - If complex → DistilBERT
        - Else → SVM+TF-IDF
    """

    # 1. Language detection
    try:
        lang = langdetect.detect(text)
    except Exception:
        lang = "unknown"

    if lang != "en":
        return "distilbert"

    # 2. Length check
    word_count = len(text.split())
    if word_count > MAX_LENGTH:
        return "distilbert"

    # 3. Complexity check using HF model
    complexity = estimate_complexity(text)

    if complexity > COMPLEXITY_THRESHOLD:
        return "distilbert"

    # Otherwise → SVM
    return "svm"


# --------------------------
# USAGE EXAMPLE
# --------------------------
if __name__ == "__main__":
    sample = "J'ai besoin d'aide pour réinitialiser mon mot de passe."
    complex_ticket = (
    "Our distributed microservices cluster is intermittently failing under high load, "
    "causing cascading timeouts in dependent services. "
    "We've observed memory leaks in the caching layer and race conditions in the database replication, "
    "and need a detailed analysis to identify the root cause and implement a resilient solution."
)

    route = route_ticket(sample)
    print(f"Router decision: {route}")
