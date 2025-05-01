#!/usr/bin/env python3

import numpy as np
from scipy import stats
import pandas as pd

def test_wilcoxon():
    print("Testing Wilcoxon test with different sample data:")
    
    # Create some test data with different distributions
    test_cases = [
        {
            'name': 'Small difference',
            'data1': [0.51, 0.52, 0.53, 0.54, 0.55],
            'data2': [0.50, 0.51, 0.52, 0.53, 0.54]
        },
        {
            'name': 'Medium difference',
            'data1': [0.51, 0.52, 0.53, 0.54, 0.55],
            'data2': [0.41, 0.42, 0.43, 0.44, 0.45]
        },
        {
            'name': 'Large difference',
            'data1': [0.51, 0.52, 0.53, 0.54, 0.55],
            'data2': [0.21, 0.22, 0.23, 0.24, 0.25]
        },
        {
            'name': 'Identical values',
            'data1': [0.51, 0.52, 0.53, 0.54, 0.55],
            'data2': [0.51, 0.52, 0.53, 0.54, 0.55]
        },
        {
            'name': 'Mixed differences',
            'data1': [0.51, 0.42, 0.63, 0.54, 0.45],
            'data2': [0.41, 0.52, 0.43, 0.64, 0.35]
        }
    ]
    
    for case in test_cases:
        data1 = np.array(case['data1'])
        data2 = np.array(case['data2'])
        
        # Calculate mean difference
        mean_diff = np.mean(data1) - np.mean(data2)
        
        # Run Wilcoxon test
        stat, p_value = stats.wilcoxon(data1, data2)
        
        print(f"\n{case['name']}:")
        print(f"Data 1: {data1}")
        print(f"Data 2: {data2}")
        print(f"Mean difference: {mean_diff:.4f}")
        print(f"Wilcoxon statistic: {stat}")
        print(f"p-value: {p_value:.6f}")

def test_with_cv_data():
    print("\n\nTesting with sample CV data structure:")
    
    # Create sample data similar to CV results
    df1 = pd.DataFrame({
        'config': [1, 1, 1, 2, 2, 2],
        'fold': [1, 2, 3, 1, 2, 3],
        'val_loss': [0.51, 0.52, 0.53, 0.41, 0.42, 0.43]
    })
    
    df2 = pd.DataFrame({
        'config': [1, 1, 1, 2, 2, 2],
        'fold': [1, 2, 3, 1, 2, 3],
        'val_loss': [0.41, 0.42, 0.43, 0.31, 0.32, 0.33]
    })
    
    # Test proper matching implementation
    for config in [1, 2]:
        # Get data for this config, indexed by fold
        data1_by_fold = df1[df1['config'] == config].set_index('fold')['val_loss']
        data2_by_fold = df2[df2['config'] == config].set_index('fold')['val_loss']
        
        # Find common folds
        common_folds = sorted(set(data1_by_fold.index).intersection(set(data2_by_fold.index)))
        
        # Extract paired data
        data1 = np.array([data1_by_fold.loc[fold] for fold in common_folds])
        data2 = np.array([data2_by_fold.loc[fold] for fold in common_folds])
        
        # Calculate Wilcoxon test
        stat, p_value = stats.wilcoxon(data1, data2)
        
        print(f"\nConfig {config}:")
        print(f"Data 1: {data1}")
        print(f"Data 2: {data2}")
        print(f"Mean difference: {np.mean(data1) - np.mean(data2):.4f}")
        print(f"Wilcoxon statistic: {stat}")
        print(f"p-value: {p_value:.6f}")

if __name__ == "__main__":
    test_wilcoxon()
    test_with_cv_data() 