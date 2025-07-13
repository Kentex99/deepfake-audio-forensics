import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Function to analyze features
def analyze_features(original_file, cloned_file):
    try:
        # Load audio files with a common sampling rate
        y_original, sr_original = librosa.load(original_file, sr=16000)
        y_cloned, sr_cloned = librosa.load(cloned_file, sr=16000)

        # Ensure both audio arrays have the same length
        min_length = min(len(y_original), len(y_cloned))
        y_original = y_original[:min_length]
        y_cloned = y_cloned[:min_length]

        # Feature extraction
        pitch_original = librosa.yin(y_original, fmin=50, fmax=500, sr=sr_original)
        pitch_cloned = librosa.yin(y_cloned, fmin=50, fmax=500, sr=sr_cloned)

        spectral_flatness_original = librosa.feature.spectral_flatness(y=y_original)
        spectral_flatness_cloned = librosa.feature.spectral_flatness(y=y_cloned)

        power_original = librosa.feature.rms(y=y_original)
        power_cloned = librosa.feature.rms(y=y_cloned)

        # Calculate differences
        pitch_mae = np.mean(np.abs(pitch_original - pitch_cloned[:len(pitch_original)]))
        spectral_flatness_diff = np.mean(
            np.abs(spectral_flatness_original - spectral_flatness_cloned[:spectral_flatness_original.shape[1]])
        )
        power_diff = np.mean(np.abs(power_original - power_cloned[:power_original.shape[1]]))

        return {
            "Pitch Mean Absolute Error (MAE)": pitch_mae,
            "Spectral Flatness Difference": spectral_flatness_diff,
            "Power Spectral Density Difference": power_diff,
        }
    except Exception as e:
        return {"Error": str(e)}

# Function to process all files in a folder
def process_folder(original_dir, cloned_dir):
    results = []
    original_files = sorted(Path(original_dir).rglob("*.wav"))
    cloned_files = sorted(Path(cloned_dir).rglob("*.wav"))

    for original_file, cloned_file in zip(original_files, cloned_files):
        analysis_result = analyze_features(original_file, cloned_file)
        analysis_result["Original File"] = original_file.name
        analysis_result["Cloned File"] = cloned_file.name
        results.append(analysis_result)

    return results

# Function to save results as CSV
def save_csv(results, output_csv_path):
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved forensic analysis to {output_csv_path}")

# Function to visualize results using Matplotlib
def visualize_with_matplotlib(results, output_dir):
    metrics = ["Pitch Mean Absolute Error (MAE)", "Spectral Flatness Difference", "Power Spectral Density Difference"]
    os.makedirs(output_dir, exist_ok=True)

    for metric in metrics:
        data = [result[metric] for result in results if metric in result and not isinstance(result[metric], str)]
        if not data:
            print(f"Metric '{metric}' is missing or empty. Skipping visualization.")
            continue

        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=20, color="blue", alpha=0.7)
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{metric.replace(' ', '_')}_matplotlib.png"))
        plt.close()
        print(f"Matplotlib plot for {metric} saved.")

# Function to visualize results using Seaborn
def visualize_with_seaborn(results, output_dir):
    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)

    for metric in ["Pitch Mean Absolute Error (MAE)", "Spectral Flatness Difference", "Power Spectral Density Difference"]:
        if metric not in df.columns or df[metric].isnull().all():
            print(f"Metric '{metric}' is missing or empty. Skipping visualization.")
            continue

        plt.figure(figsize=(10, 6))
        sns.histplot(df[metric], bins=20, kde=True, color="orange")
        plt.title(f"Distribution of {metric} (Seaborn)")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{metric.replace(' ', '_')}_seaborn.png"))
        plt.close()
        print(f"Seaborn plot for {metric} saved.")

# Main function to process all folders
def main():
    base_dir = r"E:\Research Project\cloned-original"
    output_dir = r"E:\Research Project\3. Analysis\5. Forensic Analysis"
    folders = ["001","005", "014", "015", "018", "020", "027"]

    for folder in folders:
        print(f"Processing folder {folder}...")
        original_dir = os.path.join(base_dir, folder, "original")
        cloned_dir = os.path.join(base_dir, folder, "cloned")
        folder_output_dir = os.path.join(output_dir, folder)

        # Analyze features
        results = process_folder(original_dir, cloned_dir)

        # Save results to CSV
        output_csv_path = os.path.join(output_dir, f"{folder}_forensic_analysis.csv")
        save_csv(results, output_csv_path)

        # Visualize results
        matplotlib_output_dir = os.path.join(folder_output_dir, "matplotlib")
        seaborn_output_dir = os.path.join(folder_output_dir, "seaborn")
        visualize_with_matplotlib(results, matplotlib_output_dir)
        visualize_with_seaborn(results, seaborn_output_dir)

# Run the script
if __name__ == "__main__":
    main()