from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt

# Directories
stat_dir = r"E:\Research Project\Statistical Analysis"
output_vis_dir = r"E:\Research Project\Visualize Metrics"

# Ensure output directory exists
os.makedirs(output_vis_dir, exist_ok=True)

def plot_metric(folder, metric, data, output_folder):
    """
    Plot a single metric.
    Args:
        folder (str): Folder name (for title).
        metric (str): Metric name.
        data (pd.DataFrame): DataFrame containing the metric data.
        output_folder (str): Directory to save the plot.
    """
    os.makedirs(output_folder, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.bar(data["File"], data[metric], color="skyblue")
    plt.xticks(rotation=90)
    plt.title(f"{metric} in {folder}")
    plt.ylabel(metric)
    plt.tight_layout()
    output_path = os.path.join(output_folder, f"{metric}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {metric} plot for {folder} to {output_path}")

def process_visualizations(stat_dir, output_dir):
    """
    Create visualizations for statistical metrics.
    Args:
        stat_dir (str): Directory containing statistical analysis CSV files.
        output_dir (str): Directory to save visualizations.
    """
    for csv_file in Path(stat_dir).rglob("*.csv"):
        folder = csv_file.stem.split("_")[0]
        output_folder = os.path.join(output_dir, folder)
        data = pd.read_csv(csv_file)
        for metric in ["Mean", "Variance", "Std Deviation"]:
            plot_metric(folder, metric, data, output_folder)

if __name__ == "__main__":
    process_visualizations(stat_dir, output_vis_dir)