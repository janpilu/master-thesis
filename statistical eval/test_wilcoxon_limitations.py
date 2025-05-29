#!/usr/bin/env python3
"""
Script to demonstrate the limited number of distinct p-values possible
with the Wilcoxon signed-rank test for small sample sizes.
"""

import numpy as np
from scipy.stats import wilcoxon


def enumerate_all_possible_p_values(n: int) -> set:
    """
    Enumerate all possible p-values for Wilcoxon test with n pairs.
    This is computationally intensive but demonstrates the limitation.
    """
    print(f"Enumerating all possible p-values for n={n} pairs...")

    p_values = set()

    # Generate all possible sign patterns for differences
    # Each difference can be positive (+1) or negative (-1)
    # We'll use a simplified approach: generate different rank patterns

    # For n=10, ranks are 1,2,3,4,5,6,7,8,9,10
    ranks = list(range(1, n + 1))

    # We'll test with many different difference patterns
    # to see what p-values are possible
    tested_patterns = 0

    for i in range(min(1000, 2**n)):  # Limit to avoid infinite computation
        # Create a random pattern of differences
        np.random.seed(i)

        # Generate random differences, ensuring some variation
        differences = np.random.normal(0, 1, n)

        # Create paired data where differences follow this pattern
        data1 = np.random.normal(0.5, 0.1, n)
        data2 = data1 - differences

        try:
            _, p_value = wilcoxon(data1, data2)
            p_values.add(
                round(p_value, 10)
            )  # Round to avoid floating point precision issues
            tested_patterns += 1
        except ValueError:
            # Skip if all differences are zero
            continue

    print(f"Tested {tested_patterns} different patterns")
    return p_values


def test_specific_patterns(n: int) -> dict:
    """
    Test specific patterns that should give different p-values.
    """
    patterns = {}

    # Pattern 1: All differences positive (maximum effect)
    data1 = np.arange(1, n + 1, dtype=float)
    data2 = np.zeros(n)
    _, p_val = wilcoxon(data1, data2)
    patterns["all_positive"] = p_val

    # Pattern 2: All differences negative (maximum effect, opposite direction)
    data1 = np.zeros(n)
    data2 = np.arange(1, n + 1, dtype=float)
    _, p_val = wilcoxon(data1, data2)
    patterns["all_negative"] = p_val

    # Pattern 3: Half positive, half negative (smaller effect)
    data1 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)[:n]
    data2 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)[:n]
    _, p_val = wilcoxon(data1, data2)
    patterns["mixed_half"] = p_val

    # Pattern 4: One large difference, rest small
    data1 = np.array([10] + [0.1] * (n - 1))
    data2 = np.zeros(n)
    _, p_val = wilcoxon(data1, data2)
    patterns["one_large"] = p_val

    # Pattern 5: Gradual increase
    data1 = np.linspace(0.1, 1.0, n)
    data2 = np.zeros(n)
    _, p_val = wilcoxon(data1, data2)
    patterns["gradual"] = p_val

    return patterns


def analyze_p_value_distribution():
    """
    Analyze the distribution of possible p-values for different sample sizes.
    """
    print("Wilcoxon Test P-Value Limitations Analysis")
    print("=" * 50)

    for n in [5, 6, 7, 8, 9, 10]:
        print(f"\nFor n = {n} pairs:")

        # Test specific patterns
        patterns = test_specific_patterns(n)
        print("  Specific pattern p-values:")
        for pattern_name, p_val in patterns.items():
            print(f"    {pattern_name}: {p_val:.6f}")

        # Enumerate many possible p-values (limited sampling)
        if n <= 8:  # Only for small n to avoid computation explosion
            possible_p_values = enumerate_all_possible_p_values(n)
            unique_p_values = sorted(possible_p_values)
            print(f"  Found {len(unique_p_values)} distinct p-values from sampling")

            if len(unique_p_values) <= 20:
                print(f"  All p-values: {[f'{p:.6f}' for p in unique_p_values]}")
            else:
                print(f"  Smallest 5: {[f'{p:.6f}' for p in unique_p_values[:5]]}")
                print(f"  Largest 5: {[f'{p:.6f}' for p in unique_p_values[-5:]]}")


def theoretical_analysis():
    """
    Provide theoretical analysis of Wilcoxon test limitations.
    """
    print("\nTheoretical Analysis:")
    print("=" * 30)

    for n in range(5, 11):
        # For Wilcoxon test, the test statistic W can range from 0 to n(n+1)/2
        max_W = n * (n + 1) // 2

        # The exact number of possible p-values depends on the discrete distribution
        # For small n, this is roughly 2^n different possible outcomes
        possible_outcomes = 2**n

        print(f"n = {n}:")
        print(f"  Max test statistic W: {max_W}")
        print(f"  Approximate possible outcomes: {possible_outcomes}")
        print(f"  Approximate distinct p-values: ~{possible_outcomes // 2}")

        # Minimum possible p-value (two-tailed)
        # This occurs when all differences have the same sign
        min_p = 2 * (1 / possible_outcomes)
        print(f"  Theoretical minimum p-value: {min_p:.6f}")


def demonstrate_with_cv_data():
    """
    Demonstrate the limitation using cross-validation-like data.
    """
    print("\nDemonstration with CV-like Data:")
    print("=" * 40)

    n_folds = 10
    n_configs = 3

    p_values_found = set()

    # Generate many different CV scenarios
    for scenario in range(100):
        np.random.seed(scenario)

        for config in range(n_configs):
            # Generate fine-tuned results
            finetuned = 0.75 + np.random.normal(0, 0.02, n_folds)

            # Generate frozen results (slightly worse on average)
            frozen = finetuned - np.random.normal(0.01, 0.01, n_folds)

            try:
                _, p_val = wilcoxon(finetuned, frozen)
                p_values_found.add(round(p_val, 6))
            except ValueError:
                continue

    unique_p_values = sorted(p_values_found)
    print(
        f"Found {len(unique_p_values)} distinct p-values from {100 * n_configs} tests"
    )
    print(f"Range: {min(unique_p_values):.6f} to {max(unique_p_values):.6f}")

    # Show distribution of p-values
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.hist(list(unique_p_values), bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("P-value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Possible P-values (n=10 pairs)")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        "statistical eval/wilcoxon_p_value_distribution.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved p-value distribution plot to 'wilcoxon_p_value_distribution.png'")


if __name__ == "__main__":
    analyze_p_value_distribution()
    theoretical_analysis()
    demonstrate_with_cv_data()
