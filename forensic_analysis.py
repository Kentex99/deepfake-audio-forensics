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
output_dir = r"E:\Research\output\Forensic Analysis"
os.makedirs(output_dir, exist_ok=True)

def analyze_features(original_file, comparison_file):
    """
    Analyze features between two audio files and calculate differences in Linear Scale.
    Args:
        original_file (str): Path to the original audio file.
        comparison_file (str): Path to the comparison audio file.
    Returns:
        dict: Dictionary containing calculated metrics.
    """
    try:
        # Load audio files with a common sampling rate
        y_original, sr_original = librosa.load(original_file, sr=16000)
        y_comparison, sr_comparison = librosa.load(comparison_file, sr=16000)

        # Ensure both audio arrays have the same length
        min_length = min(len(y_original), len(y_comparison))
        y_original = y_original[:min_length]
        y_comparison = y_comparison[:min_length]

        # Convert audio amplitude from dB to Linear Scale where necessary
        y_original_linear = librosa.db_to_amplitude(y_original)  # Convert to Linear
        y_comparison_linear = librosa.db_to_amplitude(y_comparison)  # Convert to Linear

        # Feature extraction in Linear Scale
        pitch_original = librosa.yin(y_original_linear, fmin=50, fmax=500, sr=sr_original)
        pitch_comparison = librosa.yin(y_comparison_linear, fmin=50, fmax=500, sr=sr_comparison)

        spectral_flatness_original = librosa.feature.spectral_flatness(y=y_original_linear)
        spectral_flatness_comparison = librosa.feature.spectral_flatness(y=y_comparison_linear)

        power_original = librosa.feature.rms(y=y_original_linear)
        power_comparison = librosa.feature.rms(y=y_comparison_linear)

        # Calculate differences in Linear Scale
        pitch_mae = np.mean(np.abs(pitch_original - pitch_comparison[:len(pitch_original)]))
        spectral_flatness_diff = np.mean(
            np.abs(spectral_flatness_original - spectral_flatness_comparison[:spectral_flatness_original.shape[1]])
        )
        power_diff = np.mean(np.abs(power_original - power_comparison[:power_original.shape[1]]))

        return {
            "Pitch Mean Absolute Error (MAE)": pitch_mae,
            "Spectral Flatness Difference": spectral_flatness_diff,
            "Power Spectral Density Difference": power_diff,
        }
    except Exception as e:
        return {"Error": str(e)}

def process_comparison(category, human_column, comparison_column):
    """
    Process comparison between human and an alternative category (cloned/anonymized/synthetic).
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

            # Analyze features
            metrics = analyze_features(human_path, comparison_path)
            metrics["Human File"] = human_file
            metrics[f"{category.capitalize()} File"] = comparison_file
            results.append(metrics)
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
    Visualize metrics using Matplotlib and Seaborn.
    Args:
        results_file (str): Path to the CSV file with analysis results.
        output_dir (str): Directory to save the visualization plots.
        category (str): Category being compared (cloned/anonymized/synthetic).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_file)

    # Metrics to visualize
    metrics = ["Pitch Mean Absolute Error (MAE)", "Spectral Flatness Difference", "Power Spectral Density Difference"]

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
    process_comparison("cloned", "human_file_name", "cloned_file_name")
    process_comparison("anonymized", "human_file_name", "anonymized_file_name")
    process_comparison("synthetic", "human_file_name", "synthetic_file_name")

    # Visualize results
    for category in ["cloned", "anonymized", "synthetic"]:
        results_csv = os.path.join(output_dir, f"human_vs_{category}", f"human_vs_{category}_analysis.csv")
        visualization_output_dir = os.path.join(output_dir, f"human_vs_{category}", "visualizations")
        visualize_metrics(results_csv, visualization_output_dir, category)