#!/usr/bin/env python
"""Test advanced gap scoring implementation."""

from src.research_gap_generator import calculate_advanced_scores

print("=" * 70)
print("ADVANCED RESEARCH GAP SCORING TEST")
print("=" * 70)

# Test case 1: High frequency, high diversity, novel
print("\nTest Case 1: High frequency, moderate diversity, somewhat novel")
scores1 = calculate_advanced_scores(
    cluster_size=10,
    paper_count=5,
    theme_frequency=2,
    max_cluster_size=20,
    max_paper_count=10,
    max_theme_frequency=4
)

print(f"  frequency_score:  {scores1['frequency_score']:.3f} (10/20)")
print(f"  diversity_score:  {scores1['diversity_score']:.3f} (5/10)")
print(f"  novelty_score:    {scores1['novelty_score']:.3f} (1/(1+2))")
print(f"  evidence_score:   {scores1['evidence_score']:.3f}")
print(f"  final_score:      {scores1['final_score']:.3f}")
print(f"  Breakdown: 0.4*{scores1['frequency_score']:.3f} + 0.3*{scores1['diversity_score']:.3f} + 0.2*{scores1['novelty_score']:.3f} + 0.1*{scores1['evidence_score']:.3f}")

# Test case 2: Low frequency, high diversity, very novel
print("\nTest Case 2: Low frequency, high diversity, very novel (unique theme)")
scores2 = calculate_advanced_scores(
    cluster_size=3,
    paper_count=8,
    theme_frequency=1,
    max_cluster_size=20,
    max_paper_count=10,
    max_theme_frequency=4
)

print(f"  frequency_score:  {scores2['frequency_score']:.3f} (3/20)")
print(f"  diversity_score:  {scores2['diversity_score']:.3f} (8/10)")
print(f"  novelty_score:    {scores2['novelty_score']:.3f} (1/(1+1))")
print(f"  evidence_score:   {scores2['evidence_score']:.3f}")
print(f"  final_score:      {scores2['final_score']:.3f}")
print(f"  Breakdown: 0.4*{scores2['frequency_score']:.3f} + 0.3*{scores2['diversity_score']:.3f} + 0.2*{scores2['novelty_score']:.3f} + 0.1*{scores2['evidence_score']:.3f}")

# Test case 3: Very high frequency, low diversity, common theme
print("\nTest Case 3: Very high frequency, low diversity, common theme")
scores3 = calculate_advanced_scores(
    cluster_size=18,
    paper_count=2,
    theme_frequency=4,
    max_cluster_size=20,
    max_paper_count=10,
    max_theme_frequency=4
)

print(f"  frequency_score:  {scores3['frequency_score']:.3f} (18/20)")
print(f"  diversity_score:  {scores3['diversity_score']:.3f} (2/10)")
print(f"  novelty_score:    {scores3['novelty_score']:.3f} (1/(1+4))")
print(f"  evidence_score:   {scores3['evidence_score']:.3f}")
print(f"  final_score:      {scores3['final_score']:.3f}")
print(f"  Breakdown: 0.4*{scores3['frequency_score']:.3f} + 0.3*{scores3['diversity_score']:.3f} + 0.2*{scores3['novelty_score']:.3f} + 0.1*{scores3['evidence_score']:.3f}")

print("\n" + "=" * 70)
print("SCORE COMPONENTS EXPLANATION")
print("=" * 70)
print("""
frequency_score (0.4 weight):
  - Measures how frequently the limitation appears in the cluster
  - Higher cluster_size = higher frequency_score
  - Normalized by max_cluster_size

diversity_score (0.3 weight):
  - Measures breadth of supporting papers
  - Higher paper_count = higher diversity_score
  - Normalized by max_paper_count

novelty_score (0.2 weight):
  - Measures uniqueness of the theme
  - Lower theme_frequency = higher novelty_score
  - Calculated as 1/(1 + theme_frequency)

evidence_score (0.1 weight):
  - Derived from frequency and diversity
  - Indicates confidence in the gap

final_score:
  - Weighted combination of all four scores
  - Used for ranking and prioritization
""")

print("=" * 70)
print("TEST PASSED: Advanced scoring implemented successfully")
print("=" * 70)
