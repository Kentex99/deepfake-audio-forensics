import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Excel file location for matching information
excel_file = r"E:\Research Project\matched_text_ids\matched_text_ids\matched_text_ids_unique.xlsx"

# Directories for input spectrograms
input_dirs = {
    "human": r"E:\Research Project\output\Spectrograms\human",
    "cloned": r"E:\Research Project\output\Spectrograms\cloned",
    "anonymized": r"E:\Research Project\output\Spectrograms\anonymized",
    "synthetic": r"E:\Research Project\output\Spectrograms\synthetic",
}

# Output directory for contrast plots
output_base_dir = r"E:\Research Project\output\Visualize Metrics\Contrast"
os.makedirs(output_base_dir, exist_ok=True)

def plot_contrast(original_path, comparison_path, output_path, title):
    """
    Compare contrast by plotting histograms of pixel intensity distributions.
    Args:
        original_path (str): Path to the human spectrogram image.
        comparison_path (str): Path to the alternative spectrogram image.
        output_path (str): Path to save the contrast comparison plot.
        title (str): Title for the comparison plot.
    """
    try:
        # Load images in grayscale
        original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        comparison_img = cv2.imread(comparison_path, cv2.IMREAD_GRAYSCALE)

        # Compute histograms
        original_hist = cv2.calcHist([original_img], [0], None, [256], [0, 256]).flatten()
        comparison_hist = cv2.calcHist([comparison_img], [0], None, [256], [0, 256]).flatten()

        # Plot histograms
        plt.figure(figsize=(10, 6))
        plt.plot(original_hist, label="Human", color="blue", linewidth=2)
        plt.plot(comparison_hist, label=title.split(" vs ")[1], color="red", linestyle="--", linewidth=2)
        plt.title(f"Pixel Intensity Distribution: {title}")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved contrast comparison plot: {output_path}")
    except Exception as e:
        print(f"Failed to process {original_path} and {comparison_path}: {e}")

def process_contrast(category, human_column, comparison_column):
    """
    Process contrast for human vs alternatives.
    Args:
        category (str): Category to compare (cloned, anonymized, synthetic).
        human_column (str): Column name for human files in the Excel.
        comparison_column (str): Column name for alternative files in the Excel.
    """
    # Read Excel file
    df = pd.read_excel(excel_file)

    # Output directory for this category
    output_dir = os.path.join(output_base_dir, f"human_vs_{category}")
    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        try:
            # Get filenames from the Excel
            human_file = row[human_column]
            comparison_file = row[comparison_column]

            # Build file paths
            human_path = os.path.join(input_dirs["human"], f"{human_file}.png")
            comparison_path = os.path.join(input_dirs[category], f"{comparison_file}.png")

            # Check if files exist
            if not (os.path.exists(human_path) and os.path.exists(comparison_path)):
                print(f"Files not found: {human_file}, {comparison_file}")
                continue

            # Generate output path and title
            output_path = os.path.join(output_dir, f"{human_file}_vs_{comparison_file}.png")
            title = f"Human vs {category.capitalize()}"

            # Plot contrast
            plot_contrast(human_path, comparison_path, output_path, title)
        except Exception as e:
            print(f"Error processing {row}: {e}")

if __name__ == "__main__":
    # Process contrast for each category
    process_contrast("cloned", "human_file_name", "cloned_file_name")
    process_contrast("anonymized", "human_file_name", "anonymized_file_name")
    process_contrast("synthetic", "human_file_name", "synthetic_file_name")
