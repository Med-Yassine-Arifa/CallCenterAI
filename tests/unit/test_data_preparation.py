import pytest
import pandas as pd
import numpy as np
import sys
sys.path.append('src/data_preparation')
from src.data_preparation.prepare_data import clean_text


class TestTextCleaning:
    """Tests pour le nettoyage de texte"""
    
    def test_clean_text_basic(self):
        """Test basique de nettoyage"""
        text = "Hello World! This is a TEST."
        result = clean_text(text)
        expected = "hello world this is a test"
        assert result == expected
    
    def test_clean_text_special_chars(self):
        """Test avec caractères spéciaux"""
        text = "Email: user@example.com & Phone: 123-456-7890"
        result = clean_text(text)
        # Doit supprimer les caractères spéciaux mais garder les espaces
        assert "@" not in result
        assert "&" not in result
        assert "-" not in result
    
    def test_clean_text_multiple_spaces(self):
        """Test avec espaces multiples"""
        text = "Too        many     spaces"
        result = clean_text(text)
        expected = "too many spaces"
        assert result == expected
    
    def test_clean_text_nan(self):
        """Test avec valeur NaN"""
        result = clean_text(np.nan)
        assert result == ""
    
    def test_clean_text_empty(self):
        """Test avec texte vide"""
        result = clean_text("")
        assert result == ""





if __name__ == "__main__":
    pytest.main([__file__, "-v"])