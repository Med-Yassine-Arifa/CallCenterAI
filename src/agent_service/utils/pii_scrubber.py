"""
Nettoyage des informations personnelles identifiables (PII)
"""
import re
from typing import List, Tuple


class PIIScrubber:
    """Classe pour supprimer les PII des textes"""

    def __init__(self):
        # Patterns regex pour détecter les PII
        self.patterns = [
            # Emails
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
            # Numéros de téléphone (formats variés)
            (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]"),
            (
                r"\b\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b",
                "[PHONE]",
            ),
            # Numéros de carte de crédit
            (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CREDIT_CARD]"),
            # Numéros de sécurité sociale (US)
            (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
            # Adresses IP
            (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP_ADDRESS]"),
            # URLs
            (r"https?://[^\s]+", "[URL]"),
        ]

    def scrub(self, text: str) -> Tuple[str, List[str]]:
        """
        Nettoyer le texte en supprimant les PII
        Args:
        text: Texte original
        Returns:
        Tuple (texte nettoyé, liste des PII trouvées)
        """
        scrubbed_text = text
        found_pii = []
        for pattern, replacement in self.patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                found_pii.extend([f"{replacement}: {len(matches)} occurrence(s)"])
                scrubbed_text = re.sub(pattern, replacement, scrubbed_text, flags=re.IGNORECASE)
        return scrubbed_text, found_pii

    def has_pii(self, text: str) -> bool:
        """Vérifier si le texte contient des PII"""
        for pattern, _ in self.patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
