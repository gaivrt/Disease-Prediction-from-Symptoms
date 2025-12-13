import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.loader import DataLoader

def analyze_interactions(top_n=20):
    # 1. Load Data
    loader = DataLoader()
    X_train, _, _, _, _ = loader.load_data()
    
    print(f"Analyzing interactions across {X_train.shape[0]} samples and {X_train.shape[1]} descriptors...")
    
    # 2. Compute Co-occurrence Matrix
    # Since data is binary (0/1), X.T @ X gives the count of co-occurrences
    # We use matrix multiplication for speed
    X_matrix = X_train.values
    co_occurrence = np.dot(X_matrix.T, X_matrix)
    
    # Fill diagonal with 0 to ignore self-occurrence (optional, but good for heatmap contrast)
    np.fill_diagonal(co_occurrence, 0)
    
    co_occ_df = pd.DataFrame(co_occurrence, index=X_train.columns, columns=X_train.columns)
    
    # 3. Find Top N Interactions
    # Unstack and sort
    unstacked = co_occ_df.unstack()
    # Remove duplicates (A-B is same as B-A) and self-pairs (if not zeroed)
    pairs = unstacked.sort_values(ascending=False)
    
    # Filter to get unique pairs (A, B) where A < B to avoid double counting
    unique_pairs = []
    seen = set()
    for (idx1, idx2), val in pairs.items():
        if idx1 == idx2: continue # Should be 0 anyway
        key = tuple(sorted((idx1, idx2)))
        if key not in seen:
            seen.add(key)
            unique_pairs.append({'Symptom_A': idx1, 'Symptom_B': idx2, 'Co-occurrence': val})
            if len(unique_pairs) >= top_n:
                break
    
    top_pairs_df = pd.DataFrame(unique_pairs)
    print("\n--- Top 10 Symptom Pairs ---")
    print(top_pairs_df.head(10))
    
    # 4. Plot Heatmap for Top Features (Using Matplotlib due to Seaborn/MPL version conflict)
    # Identify symptoms involved in top interactions to subset the heatmap
    top_symptoms = list(set(top_pairs_df['Symptom_A']).union(set(top_pairs_df['Symptom_B'])))
    subset_df = co_occ_df.loc[top_symptoms, top_symptoms]
    
    plt.figure(figsize=(10, 8))
    # matrix show
    plt.imshow(subset_df.values, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    
    # Ticks
    plt.xticks(range(len(top_symptoms)), top_symptoms, rotation=90)
    plt.yticks(range(len(top_symptoms)), top_symptoms)
    
    plt.title(f"Symptom Co-occurrence Heatmap (Top {len(top_symptoms)} Symptoms)")
    plt.tight_layout()
    
    os.makedirs('./output', exist_ok=True)
    save_path = './output/symptom_co_occurrence.png'
    plt.savefig(save_path)
    print(f"\nHeatmap saved to {save_path}")

if __name__ == "__main__":
    analyze_interactions()
