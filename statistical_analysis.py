import os
import numpy as np
import pandas as pd
from pathlib import Path

# Base directories
base_dir = r"E:\Research Project\Spectrograms"
folders = ["001", "005", "014", "015", "018", "020", "027"]
output_stat_dir = r"E:\Research Project\Statistical Analysis"

# Ensure output directory exists
os.makedirs(output_stat_dir, exist_ok=True)

def calculate_statistics(vector_files):
    """
    Calculate statistics (mean, variance, std deviation) for vector files.
    Args:
        vector_files (list): List of vector file paths.
    Returns:
        list: List of dictionaries with statistical data for each file.
    """
    statistics = []
    for vector_file in vector_files:
        try:
            data = np.load(vector_file)
            stats = {
                "File": vector_file.name,
                "Mean": np.mean(data),
                "Variance": np.var(data),
                "Std Deviation": np.std(data),
            }
            statistics.append(stats)
        except Exception as e:
            print(f"Error processing {vector_file}: {e}")
    return statistics

def process_statistics(base_dir, folders, output_dir):
    """
    Process all folders for statistical analysis.
    Args:
        base_dir (str): Base directory for spectrogram folders.
        folders (list): List of folder names to process.
        output_dir (str): Directory to save statistical analysis results.
    """
    for folder in folders:
        original_dir = os.path.join(base_dir, folder, "original", "Vectors")
        cloned_dir = os.path.join(base_dir, folder, "cloned", "Vectors")
        output_csv = os.path.join(output_dir, f"{folder}_statistics.csv")
        
        original_files = sorted(Path(original_dir).rglob("*.npy"))
        cloned_files = sorted(Path(cloned_dir).rglob("*.npy"))
        
        # Analyze both original and cloned vectors
        stats_original = calculate_statistics(original_files)
        stats_cloned = calculate_statistics(cloned_files)
        
        # Combine and save results
        df = pd.DataFrame(stats_original + stats_cloned)
        df.to_csv(output_csv, index=False)
        print(f"Saved statistical analysis to {output_csv}")

if __name__ == "__main__":
    process_statistics(base_dir, folders, output_stat_dir)