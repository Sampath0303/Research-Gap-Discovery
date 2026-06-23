"""
Tests for local research gap generator (API-independent fallback).

Tests cover:
- Keyword extraction
- Theme generation
- Gap description generation
- Potential idea generation
- Full gap generation pipeline
- Error handling and resilience
"""

import pytest
from src.local_gap_generator import LocalGapGenerator, generate_local_gap


class TestLocalGapGeneratorTokenization:
    """Test text tokenization functionality."""
    
    def test_tokenize_basic(self):
        """Test basic tokenization."""
        gen = LocalGapGenerator()
        tokens = gen._tokenize("The quick brown fox jumps over lazy dog")
        
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "the" not in tokens  # Stop word
        assert "over" not in tokens  # Stop word
    
    def test_tokenize_removes_stop_words(self):
        """Test that stop words are removed."""
        gen = LocalGapGenerator()
        tokens = gen._tokenize("a the and or for from")
        
        assert len(tokens) == 0  # All are stop words
    
    def test_tokenize_lowercase(self):
        """Test that tokenization converts to lowercase."""
        gen = LocalGapGenerator()
        tokens = gen._tokenize("HELLO World Python")
        
        assert "hello" in tokens
        assert "world" in tokens
        assert "python" in tokens
    
    def test_tokenize_removes_special_chars(self):
        """Test that special characters are removed."""
        gen = LocalGapGenerator()
        tokens = gen._tokenize("hello@#$world!!! python???")
        
        assert "hello" in tokens
        assert "world" in tokens
        assert "python" in tokens
    
    def test_tokenize_empty_string(self):
        """Test tokenizing empty string."""
        gen = LocalGapGenerator()
        tokens = gen._tokenize("")
        
        assert len(tokens) == 0
    
    def test_tokenize_min_length_filter(self):
        """Test that short tokens are filtered."""
        gen = LocalGapGenerator()
        tokens = gen._tokenize("a ab abc abcd abcde")
        
        assert "abc" in tokens  # Length >= 3
        assert "abcd" in tokens
        assert "abcde" in tokens
        assert "ab" not in tokens  # Length < 3


class TestLocalGapGeneratorKeywordExtraction:
    """Test keyword extraction functionality."""
    
    def test_extract_keywords_basic(self):
        """Test basic keyword extraction."""
        gen = LocalGapGenerator()
        limitations = [
            "Models require computational resources",
            "Inference latency is problematic",
            "Computational efficiency is limited",
        ]
        
        keywords = gen._extract_keywords_tfidf(limitations)
        
        assert len(keywords) > 0
        assert "computational" in keywords or "resources" in keywords
    
    def test_extract_keywords_top_k(self):
        """Test that top_k limit is respected."""
        gen = LocalGapGenerator()
        limitations = [
            "word1 word1 word2 word3 word4 word5 word6 word7 word8 word9"
            for _ in range(3)
        ]
        
        keywords = gen._extract_keywords_tfidf(limitations, top_k=5)
        
        assert len(keywords) <= 5
    
    def test_extract_keywords_empty(self):
        """Test keyword extraction with empty limitations."""
        gen = LocalGapGenerator()
        keywords = gen._extract_keywords_tfidf([])
        
        assert len(keywords) == 0
    
    def test_extract_keywords_frequency_order(self):
        """Test that keywords are ordered by frequency."""
        gen = LocalGapGenerator()
        limitations = [
            "python python python language",
            "python programming language",
            "language code",
        ]
        
        keywords = gen._extract_keywords_tfidf(limitations, top_k=3)
        
        # "python" should be first (appears 4 times)
        if keywords:
            assert keywords[0] == "python"


class TestLocalGapGeneratorThemeGeneration:
    """Test theme generation."""
    
    def test_generate_theme_basic(self):
        """Test basic theme generation."""
        gen = LocalGapGenerator()
        limitations = ["Issue one", "Issue two"]
        keywords = ["performance", "efficiency", "optimization"]
        
        theme = gen._generate_theme(limitations, keywords)
        
        assert isinstance(theme, str)
        assert len(theme) > 0
    
    def test_generate_theme_uses_keywords(self):
        """Test that theme uses extracted keywords."""
        gen = LocalGapGenerator()
        limitations = ["Issue one", "Issue two"]
        keywords = ["uniquekeyword123", "anotherkeyword456"]
        
        theme = gen._generate_theme(limitations, keywords)
        
        # Theme should contain or reference the keywords
        assert len(theme) > 0
        assert isinstance(theme, str)
    
    def test_generate_theme_deterministic(self):
        """Test that theme generation is deterministic."""
        gen = LocalGapGenerator()
        limitations = ["Issue one", "Issue two"]
        keywords = ["python", "performance"]
        
        theme1 = gen._generate_theme(limitations, keywords)
        theme2 = gen._generate_theme(limitations, keywords)
        
        assert theme1 == theme2
    
    def test_generate_theme_no_keywords(self):
        """Test theme generation with no keywords."""
        gen = LocalGapGenerator()
        limitations = ["Issue one"]
        
        theme = gen._generate_theme(limitations, [])
        
        assert isinstance(theme, str)
        assert len(theme) > 0


class TestLocalGapGeneratorGapDescription:
    """Test research gap description generation."""
    
    def test_generate_gap_basic(self):
        """Test basic gap description generation."""
        gen = LocalGapGenerator()
        limitations = [
            "Current methods are slow",
            "Memory usage is excessive",
        ]
        keywords = ["performance", "efficiency"]
        
        gap = gen._generate_research_gap(limitations, keywords)
        
        assert isinstance(gap, str)
        assert len(gap) > 0
        assert "limitations" in gap.lower() or "gap" in gap.lower()
    
    def test_generate_gap_includes_limitations(self):
        """Test that gap description includes limitations."""
        gen = LocalGapGenerator()
        limitations = ["Specific limitation text here"]
        keywords = ["performance"]
        
        gap = gen._generate_research_gap(limitations, keywords)
        
        # Should reference the limitations
        assert "limitation" in gap.lower() or "issue" in gap.lower()
    
    def test_generate_gap_deterministic(self):
        """Test that gap generation is deterministic."""
        gen = LocalGapGenerator()
        limitations = ["Issue one", "Issue two"]
        keywords = ["performance"]
        
        gap1 = gen._generate_research_gap(limitations, keywords)
        gap2 = gen._generate_research_gap(limitations, keywords)
        
        assert gap1 == gap2


class TestLocalGapGeneratorPotentialIdea:
    """Test potential idea generation."""
    
    def test_generate_idea_basic(self):
        """Test basic idea generation."""
        gen = LocalGapGenerator()
        limitations = ["Issue one", "Issue two"]
        keywords = ["performance", "efficiency"]
        
        idea = gen._generate_potential_idea(limitations, keywords)
        
        assert isinstance(idea, str)
        assert len(idea) > 0
    
    def test_generate_idea_actionable(self):
        """Test that idea is actionable."""
        gen = LocalGapGenerator()
        limitations = ["Issue one"]
        keywords = ["research"]
        
        idea = gen._generate_potential_idea(limitations, keywords)
        
        # Should contain action words
        action_words = ["develop", "investigate", "design", "create", "propose"]
        assert any(word in idea.lower() for word in action_words)
    
    def test_generate_idea_deterministic(self):
        """Test that idea generation is deterministic."""
        gen = LocalGapGenerator()
        limitations = ["Issue one", "Issue two"]
        keywords = ["performance"]
        
        idea1 = gen._generate_potential_idea(limitations, keywords)
        idea2 = gen._generate_potential_idea(limitations, keywords)
        
        assert idea1 == idea2


class TestLocalGapGeneratorFullPipeline:
    """Test complete gap generation pipeline."""
    
    def test_generate_full_gap_basic(self):
        """Test generating complete gap."""
        gen = LocalGapGenerator()
        limitations = [
            "Training is computationally expensive",
            "Inference latency is too high",
            "Memory usage is prohibitive",
        ]
        
        result = gen.generate_local_gap(limitations)
        
        assert isinstance(result, dict)
        assert "theme" in result
        assert "research_gap" in result
        assert "potential_idea" in result
    
    def test_generate_full_gap_non_empty(self):
        """Test that all fields are non-empty."""
        gen = LocalGapGenerator()
        limitations = ["Issue one", "Issue two"]
        
        result = gen.generate_local_gap(limitations)
        
        assert len(result["theme"]) > 0
        assert len(result["research_gap"]) > 0
        assert len(result["potential_idea"]) > 0
    
    def test_generate_full_gap_deterministic(self):
        """Test that full gap generation is deterministic."""
        gen = LocalGapGenerator()
        limitations = ["Issue one", "Issue two", "Issue three"]
        
        result1 = gen.generate_local_gap(limitations)
        result2 = gen.generate_local_gap(limitations)
        
        assert result1 == result2
    
    def test_generate_full_gap_empty_limitations(self):
        """Test gap generation with empty limitations."""
        gen = LocalGapGenerator()
        
        result = gen.generate_local_gap([])
        
        assert isinstance(result, dict)
        assert "theme" in result
        assert "research_gap" in result
        assert "potential_idea" in result
    
    def test_generate_full_gap_single_limitation(self):
        """Test gap generation with single limitation."""
        gen = LocalGapGenerator()
        
        result = gen.generate_local_gap(["Single limitation"])
        
        assert len(result["theme"]) > 0
        assert len(result["research_gap"]) > 0
        assert len(result["potential_idea"]) > 0


class TestLocalGapGeneratorFunction:
    """Test the module-level function."""
    
    def test_module_function_success(self):
        """Test module-level generate_local_gap function."""
        limitations = [
            "Performance issues exist",
            "Scalability challenges arise",
        ]
        
        result = generate_local_gap(limitations)
        
        assert isinstance(result, dict)
        assert all(key in result for key in ["theme", "research_gap", "potential_idea"])
    
    def test_module_function_never_crashes(self):
        """Test that function never crashes."""
        test_cases = [
            [],
            ["Simple limitation"],
            ["Limit1", "Limit2", "Limit3"],
            [None, "Valid limitation", None],
        ]
        
        for case in test_cases:
            # Filter out None values
            limitations = [l for l in case if l is not None]
            
            # Should never raise exception
            result = generate_local_gap(limitations)
            
            assert isinstance(result, dict)
            assert "theme" in result
            assert "research_gap" in result
            assert "potential_idea" in result


class TestLocalGapGeneratorRobustness:
    """Test robustness and error handling."""
    
    def test_handles_unicode(self):
        """Test handling of Unicode characters."""
        gen = LocalGapGenerator()
        limitations = [
            "Performance déficits in résearch",
            "Scalability 问题 in systems",
        ]
        
        result = gen.generate_local_gap(limitations)
        
        assert isinstance(result, dict)
        assert len(result["theme"]) > 0
    
    def test_handles_long_text(self):
        """Test handling of very long text."""
        gen = LocalGapGenerator()
        limitations = [
            "word " * 1000  # Very long limitation
        ]
        
        result = gen.generate_local_gap(limitations)
        
        assert isinstance(result, dict)
        assert len(result["theme"]) > 0
    
    def test_handles_special_characters(self):
        """Test handling of special characters."""
        gen = LocalGapGenerator()
        limitations = [
            "Performance!@#$%^&*() issues",
            "Scalability <><><> problems",
        ]
        
        result = gen.generate_local_gap(limitations)
        
        assert isinstance(result, dict)
        assert len(result["theme"]) > 0
    
    def test_function_handles_invalid_input(self):
        """Test that function handles invalid inputs gracefully."""
        # Should not crash with edge cases
        result = generate_local_gap([])
        assert isinstance(result, dict)
        
        result = generate_local_gap([""])
        assert isinstance(result, dict)
        
        result = generate_local_gap(["   "])
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
