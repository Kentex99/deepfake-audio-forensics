import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Input directories for audio files
input_dirs = {
    "human": r"E:\Research\matched_text_ids\matched_text_ids\human",
    "cloned": r"E:\Research\matched_text_ids\matched_text_ids\cloned",
    "anonymized": r"E:\Research\matched_text_ids\matched_text_ids\anonymized",
    "synthetic": r"E:\Research\matched_text_ids\matched_text_ids\synthetic",
}

# Path for the Excel file
excel_file = r"E:\Research\matched_text_ids\matched_text_ids\matched_text_ids_unique.xlsx"

# Output directory
output_dir = r"E:\Research\output\Advanced Forensic Analysis"
os.makedirs(output_dir, exist_ok=True)

def extract_features(audio_file):
    """
    Extracts various speech features from an audio file.
    Args:
        audio_file (str): Path to the audio file.
    Returns:
        dict: Extracted feature values.
    """
    try:
        y, sr = librosa.load(audio_file, sr=16000)

        # Spectral Features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        energy_entropy = np.mean(librosa.feature.spectral_flatness(y=y))

        # Pitch (Fundamental Frequency Variation)
        f0_variation = np.var(librosa.yin(y, fmin=50, fmax=500, sr=sr))

        # Formant Frequencies (Approximation)
        formants = librosa.lpc(y, order=2) if len(y) > 2 else [0, 0, 0]
        f1, f2, f3 = formants[0], formants[1], formants[2] if len(formants) > 2 else 0

        # Wavelet Energy
        wavelet_energy = np.sum(np.abs(librosa.cqt(y, sr=sr))) if len(y) > 0 else 0

        return {
            "Spectral Centroid": spectral_centroid,
            "Spectral Bandwidth": spectral_bandwidth,
            "Energy Entropy": energy_entropy,
            "F0 Variation": f0_variation,
            "Formant F1": f1,
            "Formant F2": f2,
            "Formant F3": f3,
            "Wavelet Energy": wavelet_energy,
        }
    except Exception as e:
        return {"Error": str(e)}

def process_comparison(category, human_column, comparison_column):
    """
    Processes comparisons between human and an alternative category (cloned/anonymized/synthetic).
    Args:
        category (str): Category to compare (cloned, anonymized, synthetic).
        human_column (str): Column name for human files in the Excel.
        comparison_column (str): Column name for alternative files in the Excel.
    """
    # Read the Excel file
    df = pd.read_excel(excel_file)

    # Create output directories
    category_output_dir = os.path.join(output_dir, f"human_vs_{category}")
    os.makedirs(category_output_dir, exist_ok=True)

    # Results storage
    results = []
    error_log = []

    for _, row in df.iterrows():
        try:
            # File paths
            human_file = row[human_column]
            comparison_file = row[comparison_column]

            human_path = os.path.join(input_dirs["human"], f"{human_file}.wav")
            comparison_path = os.path.join(input_dirs[category], f"{comparison_file}.wav")

            # Skip if files are missing
            if not (os.path.exists(human_path) and os.path.exists(comparison_path)):
                error_log.append(f"Files not found: {human_file}, {comparison_file}")
                continue

            # Extract features for human and comparison files
            human_features = extract_features(human_path)
            comparison_features = extract_features(comparison_path)

            # Compute differences
            differences = {
                "Spectral Centroid Diff": abs(human_features["Spectral Centroid"] - comparison_features["Spectral Centroid"]),
                "Spectral Bandwidth Diff": abs(human_features["Spectral Bandwidth"] - comparison_features["Spectral Bandwidth"]),
                "Energy Entropy Diff": abs(human_features["Energy Entropy"] - comparison_features["Energy Entropy"]),
                "F0 Variation Diff": abs(human_features["F0 Variation"] - comparison_features["F0 Variation"]),
                "Formant F1 Diff": abs(human_features["Formant F1"] - comparison_features["Formant F1"]),
                "Formant F2 Diff": abs(human_features["Formant F2"] - comparison_features["Formant F2"]),
                "Formant F3 Diff": abs(human_features["Formant F3"] - comparison_features["Formant F3"]),
                "Wavelet Energy Diff": abs(human_features["Wavelet Energy"] - comparison_features["Wavelet Energy"]),
            }

            # Store results
            results.append({
                "Human File": human_file,
                "Comparison File": comparison_file,
                **human_features,
                **comparison_features,
                **differences,
            })
        except Exception as e:
            error_log.append(f"Error processing {human_file} and {comparison_file}: {e}")

    # Save results to CSV
    if results:
        output_csv = os.path.join(category_output_dir, f"human_vs_{category}_analysis.csv")
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"Saved analysis results to {output_csv}")

    # Save error log
    if error_log:
        error_log_file = os.path.join(category_output_dir, "error_log.txt")
        with open(error_log_file, "w") as log:
            log.write("\n".join(error_log))
        print(f"Saved error log to {error_log_file}")

def visualize_metrics(results_file, output_dir, category):
    """
    Visualizes metrics using Matplotlib and Seaborn.
    Args:
        results_file (str): Path to the CSV file with analysis results.
        output_dir (str): Directory to save the visualization plots.
        category (str): Category being compared (cloned/anonymized/synthetic).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_file)

    # Metrics to visualize
    metrics = ["Spectral Centroid Diff", "Spectral Bandwidth Diff", "Energy Entropy Diff", 
               "F0 Variation Diff", "Formant F1 Diff", "Formant F2 Diff", "Formant F3 Diff", "Wavelet Energy Diff"]

    for metric in metrics:
        if metric not in df.columns or df[metric].isnull().all():
            print(f"Metric '{metric}' is missing or empty. Skipping visualization.")
            continue

        # Matplotlib visualization
        plt.figure(figsize=(10, 6))
        plt.hist(df[metric], bins=20, color="blue", alpha=0.7)
        plt.title(f"Distribution of {metric}: Human vs {category.capitalize()}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{metric.replace(' ', '_')}_matplotlib.png"))
        plt.close()

        # Seaborn visualization
        plt.figure(figsize=(10, 6))
        sns.histplot(df[metric], kde=True, bins=20, color="orange")
        plt.title(f"Distribution of {metric} (Seaborn): Human vs {category.capitalize()}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{metric.replace(' ', '_')}_seaborn.png"))
        plt.close()

if __name__ == "__main__":
    # Process comparisons
    for category in ["cloned", "anonymized", "synthetic"]:
        process_comparison(category, "human_file_name", f"{category}_file_name")

    # Visualize results
    for category in ["cloned", "anonymized", "synthetic"]:
        results_csv = os.path.join(output_dir, f"human_vs_{category}", f"human_vs_{category}_analysis.csv")
        visualization_output_dir = os.path.join(output_dir, f"human_vs_{category}", "visualizations")
        visualize_metrics(results_csv, visualization_output_dir, category)
