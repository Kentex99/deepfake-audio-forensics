import os
import numpy as np
import pandas as pd
from pathlib import Path

# Base directory for spectrogram folders
base_dir = r"E:\Research Project\Spectrograms"

# Folders to process
folders = ["001", "005", "014", "015", "018", "020", "027"]

# Output directory for numerical analysis
output_base_dir = r"E:\Research Project\Numerical Analysis"
os.makedirs(output_base_dir, exist_ok=True)

def reshape_vectors(original_vector, cloned_vector):
    """
    Reshape or pad vectors to the same size for comparison.
    Args:
        original_vector (np.array): Original voice vector.
        cloned_vector (np.array): Cloned voice vector.
    Returns:
        tuple: Reshaped or padded vectors.
    """
    original_size = original_vector.size
    cloned_size = cloned_vector.size
    
    # Find the larger size
    max_size = max(original_size, cloned_size)
    
    # Pad vectors with zeros to match the larger size
    original_padded = np.pad(original_vector, (0, max_size - original_size), mode='constant')
    cloned_padded = np.pad(cloned_vector, (0, max_size - cloned_size), mode='constant')
    
    return original_padded, cloned_padded

def calculate_metrics(original_vector, cloned_vector):
    """
    Calculate numerical metrics between two vectors.
    Args:
        original_vector (np.array): Original voice vector.
        cloned_vector (np.array): Cloned voice vector.
    Returns:
        dict: Dictionary of calculated metrics.
    """
    # Reshape vectors to be compatible
    original_vector, cloned_vector = reshape_vectors(original_vector.flatten(), cloned_vector.flatten())
    
    # Metrics
    metrics = {
        "Euclidean Distance": np.linalg.norm(original_vector - cloned_vector),
        "Mean Squared Error (MSE)": np.mean((original_vector - cloned_vector) ** 2),
        "Cosine Similarity": np.dot(original_vector, cloned_vector) / (
            np.linalg.norm(original_vector) * np.linalg.norm(cloned_vector)
        ),
    }
    return metrics

def process_folder(original_dir, cloned_dir, output_dir):
    """
    Process a single folder and compute metrics between original and cloned vectors.
    Args:
        original_dir (str): Path to the original voice vectors.
        cloned_dir (str): Path to the cloned voice vectors.
        output_dir (str): Path to save the numerical analysis results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sorted lists of vector files
    original_files = sorted(Path(original_dir).rglob("*.npy"))
    cloned_files = sorted(Path(cloned_dir).rglob("*.npy"))
    
    # Pair files based on sorted order
    pairs = zip(original_files, cloned_files)
    
    # Store results
    results = []
    error_log = []

    for original_file, cloned_file in pairs:
        try:
            # Load vectors
            original_vector = np.load(original_file)
            cloned_vector = np.load(cloned_file)
            
            # Calculate metrics
            metrics = calculate_metrics(original_vector, cloned_vector)
            metrics["Original File"] = original_file.name
            metrics["Cloned File"] = cloned_file.name
            
            # Append to results
            results.append(metrics)
        except Exception as e:
            error_log.append(f"Failed to process {original_file} and {cloned_file}: {e}")
            continue
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        output_csv = os.path.join(output_dir, "numerical_analysis.csv")
        df.to_csv(output_csv, index=False)
        print(f"Saved numerical analysis to {output_csv}")
    
    # Save error log
    if error_log:
        error_log_file = os.path.join(output_dir, "error_log.txt")
        with open(error_log_file, "w") as log_file:
            log_file.write("\n".join(error_log))
        print(f"Saved error log to {error_log_file}")

def process_all_folders(base_dir, folders, output_base_dir):
    """
    Process all specified folders for numerical analysis.
    Args:
        base_dir (str): Base directory containing spectrogram folders.
        folders (list): List of folder names to process.
        output_base_dir (str): Base directory to save numerical analysis results.
    """
    for folder in folders:
        original_dir = os.path.join(base_dir, folder, "original", "Vectors")
        cloned_dir = os.path.join(base_dir, folder, "cloned", "Vectors")
        output_dir = os.path.join(output_base_dir, folder)
        
        process_folder(original_dir, cloned_dir, output_dir)

if __name__ == "__main__":
    process_all_folders(base_dir, folders, output_base_dir)