"""Exploratory Data Analysis for the Stormfront dataset."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from collections import Counter
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Set up plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

def load_stormfront_data():
    """Load the Stormfront dataset and convert to pandas DataFrame."""
    print("Loading Stormfront dataset...")
    dataset = load_dataset("odegiber/hate_speech18", trust_remote_code=True)['train']
    df = dataset.to_pandas()
    print(f"Dataset loaded with {len(df)} samples")
    return df

def analyze_class_distribution(df):
    """Analyze and visualize class distribution."""
    print("\n=== Label Distribution ===")
    label_counts = df['label'].value_counts().reset_index()
    label_counts.columns = ['Label', 'Count']
    
    # Print counts and percentages
    total = len(df)
    for idx, row in label_counts.iterrows():
        print(f"{row['Label']}: {row['Count']} samples ({row['Count']/total*100:.2f}%)")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Label', y='Count', data=label_counts)
    plt.title('Label Distribution in Stormfront Dataset')
    plt.xlabel('Label')
    plt.ylabel('Count')
    
    # Add count labels on top of bars
    for p, label in zip(ax.patches, label_counts['Count']):
        ax.annotate(format(label, ','), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig('stormfront_label_distribution.png')
    plt.show()
    
    # Create pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(label_counts['Count'], labels=label_counts['Label'], autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Label Distribution (%) in Stormfront Dataset')
    plt.tight_layout()
    plt.savefig('stormfront_label_pie.png')
    plt.show()
    
    return label_counts

def analyze_text_length(df):
    """Analyze and visualize text length distribution."""
    print("\n=== Text Length Analysis ===")
    df['text_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    
    # Summary statistics
    print("Text length statistics:")
    print(df['text_length'].describe())
    print("\nWord count statistics:")
    print(df['word_count'].describe())
    
    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Text length histogram
    sns.histplot(data=df, x='text_length', hue='label', multiple='stack', ax=axes[0])
    axes[0].set_title('Distribution of Text Lengths')
    axes[0].set_xlabel('Text Length (characters)')
    axes[0].set_ylabel('Count')
    
    # Word count histogram
    sns.histplot(data=df, x='word_count', hue='label', multiple='stack', ax=axes[1])
    axes[1].set_title('Distribution of Word Counts')
    axes[1].set_xlabel('Word Count')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('stormfront_text_length_analysis.png')
    plt.show()
    
    return df

def analyze_word_frequencies(df, label=None):
    """Analyze most common words, optionally filtered by label."""
    print(f"\n=== Word Frequency Analysis {f'for {label}' if label else ''} ===")
    
    if label:
        texts = df[df['label'] == label]['text'].tolist()
    else:
        texts = df['text'].tolist()
    
    # Create a CountVectorizer to get word frequencies
    vectorizer = CountVectorizer(stop_words='english', max_features=100)
    X = vectorizer.fit_transform(texts)
    
    # Get the word counts
    word_counts = np.sum(X.toarray(), axis=0)
    word_freq = pd.DataFrame({
        'word': vectorizer.get_feature_names_out(),
        'frequency': word_counts
    }).sort_values(by='frequency', ascending=False)
    
    # Print top 20 words
    print(f"Top 20 words {f'in {label}' if label else ''}:")
    print(word_freq.head(20))
    
    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          max_words=100, contour_width=3, contour_color='steelblue')
    wordcloud.generate_from_frequencies({word: freq for word, freq in 
                                        zip(word_freq['word'], word_freq['frequency'])})
    
    # Plot word cloud
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud {f"for {label}" if label else ""}')
    plt.tight_layout()
    plt.savefig(f'stormfront_wordcloud{"_" + label if label else ""}.png')
    plt.show()
    
    # Plot top words
    plt.figure(figsize=(12, 8))
    sns.barplot(x='frequency', y='word', data=word_freq.head(20))
    plt.title(f'Top 20 Words {f"in {label}" if label else ""}')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.tight_layout()
    plt.savefig(f'stormfront_top_words{"_" + label if label else ""}.png')
    plt.show()
    
    return word_freq

def analyze_context_distribution(df):
    """Analyze the distribution of context counts."""
    print("\n=== Context Count Analysis ===")
    
    context_counts = df['num_contexts'].value_counts().sort_index().reset_index()
    context_counts.columns = ['Context Count', 'Frequency']
    
    print("Context count distribution:")
    print(context_counts)
    
    # Plot context count distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Context Count', y='Frequency', data=context_counts)
    plt.title('Number of Contexts Distribution')
    plt.xlabel('Number of Contexts')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('stormfront_context_distribution.png')
    plt.show()
    
    # Relationship between context count and label
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='label', y='num_contexts', data=df)
    plt.title('Number of Contexts by Label')
    plt.xlabel('Label')
    plt.ylabel('Number of Contexts')
    plt.tight_layout()
    plt.savefig('stormfront_context_by_label.png')
    plt.show()
    
    return context_counts

def analyze_class_imbalance_impact(df):
    """Analyze impact of class imbalance on model performance."""
    print("\n=== Class Imbalance Impact Analysis ===")
    
    # Create binary labels (hate vs not hate)
    df['binary_label'] = df['label'].apply(lambda x: 1 if x == 'hate' else 0)
    
    # Calculate class imbalance metrics
    total = len(df)
    hate_count = sum(df['binary_label'])
    nonhate_count = total - hate_count
    
    # Calculate imbalance ratio
    imbalance_ratio = max(hate_count, nonhate_count) / min(hate_count, nonhate_count)
    
    print(f"Binary class distribution:")
    print(f"- Hate speech: {hate_count} samples ({hate_count/total*100:.2f}%)")
    print(f"- Non-hate speech: {nonhate_count} samples ({nonhate_count/total*100:.2f}%)")
    print(f"- Imbalance ratio: {imbalance_ratio:.2f}")
    
    # Calculate potential metrics for a baseline model always predicting majority class
    majority_accuracy = max(hate_count, nonhate_count) / total
    print(f"\nBaseline metrics (always predicting majority class):")
    print(f"- Accuracy: {majority_accuracy:.4f}")
    
    # Create bar chart for binary distribution
    plt.figure(figsize=(8, 6))
    binary_counts = df['binary_label'].value_counts().reset_index()
    binary_counts.columns = ['Binary Label', 'Count']
    binary_counts['Binary Label'] = binary_counts['Binary Label'].map({0: 'Non-hate', 1: 'Hate'})
    
    ax = sns.barplot(x='Binary Label', y='Count', data=binary_counts)
    plt.title('Binary Label Distribution (Hate vs Non-hate)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Add count labels on top of bars
    for p, label in zip(ax.patches, binary_counts['Count']):
        ax.annotate(format(label, ','), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig('stormfront_binary_distribution.png')
    plt.show()
    
    return {'hate_count': hate_count, 'nonhate_count': nonhate_count, 'imbalance_ratio': imbalance_ratio}

def main():
    # Load dataset
    df = load_stormfront_data()
    
    # Print basic information
    print("\n=== Dataset Overview ===")
    print(df.info())
    print("\nSample data:")
    print(df.head())
    
    # Analyze label distribution
    label_counts = analyze_class_distribution(df)
    
    # Analyze text length
    df = analyze_text_length(df)
    
    # Analyze word frequencies (overall and by label)
    word_freq = analyze_word_frequencies(df)
    for label in df['label'].unique():
        analyze_word_frequencies(df, label)
    
    # Analyze context counts
    analyze_context_distribution(df)
    
    # Analyze class imbalance for binary classification
    analyze_class_imbalance_impact(df)
    
    print("\nEDA completed. Visualizations saved to current directory.")

if __name__ == "__main__":
    main() 