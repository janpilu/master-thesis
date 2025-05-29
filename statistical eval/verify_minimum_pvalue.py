#!/usr/bin/env python3
"""
Script to verify the minimum p-value formula for Wilcoxon signed-rank test.
"""

import numpy as np
from scipy.stats import wilcoxon


def theoretical_minimum_pvalue(n: int) -> float:
    """Calculate theoretical minimum p-value for n pairs."""
    return 1 / (2 ** (n - 1))


def verify_minimum_pvalue(n: int = 10) -> None:
    """Verify the minimum p-value by creating extreme cases."""
    print(f"Verifying minimum p-value for n = {n} pairs")
    print("=" * 50)

    # Theoretical minimum
    theoretical_min = theoretical_minimum_pvalue(n)
    print(f"Theoretical minimum p-value: {theoretical_min:.10f}")
    print(f"As fraction: 1/{2 ** (n - 1)} = 1/{2 ** (n - 1)}")

    # Case 1: All differences positive (ascending order)
    data1 = np.arange(1, n + 1, dtype=float)
    data2 = np.zeros(n)

    _, p_val1 = wilcoxon(data1, data2)
    print("\nCase 1 - All positive differences:")
    print(f"Data1: {data1}")
    print(f"Data2: {data2}")
    print(f"Differences: {data1 - data2}")
    print(f"P-value: {p_val1:.10f}")

    # Case 2: All differences negative
    _, p_val2 = wilcoxon(data2, data1)
    print("\nCase 2 - All negative differences:")
    print(f"Data1: {data2}")
    print(f"Data2: {data1}")
    print(f"Differences: {data2 - data1}")
    print(f"P-value: {p_val2:.10f}")

    # Verification
    print("\nVerification:")
    print(f"Theoretical: {theoretical_min:.10f}")
    print(f"Empirical 1: {p_val1:.10f}")
    print(f"Empirical 2: {p_val2:.10f}")
    print(
        f"Match: {np.isclose(theoretical_min, p_val1) and np.isclose(theoretical_min, p_val2)}"
    )


def show_formula_for_different_n() -> None:
    """Show the minimum p-value formula for different values of n."""
    print("\nMinimum p-values for different sample sizes:")
    print("=" * 50)
    print("n\t2^(n-1)\tMin p-value\tDecimal")
    print("-" * 50)

    for n in range(3, 21):
        denominator = 2 ** (n - 1)
        min_p = 1 / denominator
        print(f"{n}\t{denominator}\t\t1/{denominator}\t\t{min_p:.8f}")


if __name__ == "__main__":
    # Verify for n=10 (your case)
    verify_minimum_pvalue(10)

    # Show formula for different n values
    show_formula_for_different_n()

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    print("For n=10 paired observations:")
    print("• Minimum possible p-value = 1/2^9 = 1/512 = 0.001953125")
    print("• Your observed p-value of 0.001953 IS the absolute minimum!")
    print("• This means ALL 10 folds showed differences in the same direction")
    print("• This is the strongest possible evidence of a systematic difference")
