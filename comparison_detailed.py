import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse

# Base directories
base_dir = r"E:\Research Project\2. Spectrograms"
output_dir = r"E:\Research Project\New folder"

# Folder structure to process
folders = ["001", "005", "014", "015", "018", "020", "027"]

def generate_colored_combined_image(original_path, cloned_path, output_path):
    """
    Generate a combined image with original, cloned, and difference spectrograms with adjusted contrast markings.
    """
    try:
        # Load images in color
        original_img = cv2.imread(original_path)
        cloned_img = cv2.imread(cloned_path)

        # Validate images are loaded
        if original_img is None or cloned_img is None:
            raise ValueError("One or both images could not be loaded.")

        # Resize cloned image to match original if needed
        if original_img.shape != cloned_img.shape:
            cloned_img = cv2.resize(cloned_img, (original_img.shape[1], original_img.shape[0]))

        # Compute absolute difference
        diff_img = cv2.absdiff(original_img, cloned_img)

        # Highlight differences with colormap
        diff_gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
        diff_highlight = cv2.merge((thresholded, thresholded, np.zeros_like(thresholded)))

        # Create a combined plot
        plt.figure(figsize=(15, 5))

        # Original spectrogram
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title("Original Spectrogram")
        plt.axis("off")

        # Cloned spectrogram
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(cloned_img, cv2.COLOR_BGR2RGB))
        plt.title("Cloned Spectrogram")
        plt.axis("off")

        # Difference spectrogram with custom markings
        plt.subplot(1, 3, 3)
        combined_diff = cv2.addWeighted(cv2.cvtColor(diff_gray, cv2.COLOR_GRAY2BGR), 0.7, diff_highlight, 0.3, 0)
        plt.imshow(combined_diff, cmap='gray')
        plt.title("Difference with Contrast Markings")
        plt.axis("off")

        # Save the combined image
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved comparison image: {output_path}")
    except Exception as e:
        print(f"Failed to process {original_path} and {cloned_path}: {e}")

def process_all_folders(base_dir, folders, output_dir):
    """
    Process all specified folders for original and cloned spectrogram comparisons.
    """
    for folder in folders:
        print(f"Processing folder: {folder}")
        original_dir = os.path.join(base_dir, folder, "original")
        cloned_dir = os.path.join(base_dir, folder, "cloned")
        folder_output_dir = os.path.join(output_dir, folder)
        os.makedirs(folder_output_dir, exist_ok=True)

        # Get sorted lists of files
        original_files = sorted(Path(original_dir).rglob("*.png"))
        cloned_files = sorted(Path(cloned_dir).rglob("*.png"))

        if not original_files:
            print(f"No spectrograms found in the original folder: {original_dir}. Skipping...")
            continue
        if not cloned_files:
            print(f"No spectrograms found in the cloned folder: {cloned_dir}. Skipping...")
            continue

        print(f"Found {len(original_files)} original and {len(cloned_files)} cloned files.")

        summary_data = []
        for original_file, cloned_file in zip(original_files, cloned_files):
            output_path = os.path.join(folder_output_dir, f"{original_file.stem}_comparison.png")
            try:
                # Generate combined image
                generate_colored_combined_image(str(original_file), str(cloned_file), output_path)

                # Calculate SSIM and MSE
                original_img = cv2.imread(str(original_file))
                cloned_img = cv2.imread(str(cloned_file))
                if original_img.shape != cloned_img.shape:
                    cloned_img = cv2.resize(cloned_img, (original_img.shape[1], original_img.shape[0]))
                original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                cloned_gray = cv2.cvtColor(cloned_img, cv2.COLOR_BGR2GRAY)
                ssim_value, _ = ssim(original_gray, cloned_gray, full=True)
                mse_value = mse(original_gray, cloned_gray)

                # Append to summary
                summary_data.append({
                    "Original File": original_file.name,
                    "Cloned File": cloned_file.name,
                    "SSIM": ssim_value,
                    "MSE": mse_value
                })
            except Exception as e:
                print(f"Error processing {original_file} and {cloned_file}: {e}")
                summary_data.append({
                    "Original File": original_file.name,
                    "Cloned File": cloned_file.name,
                    "Error": str(e)
                })

        # Save folder-specific CSV
        summary_csv_path = os.path.join(folder_output_dir, f"{folder}_analysis.csv")
        pd.DataFrame(summary_data).to_csv(summary_csv_path, index=False)
        print(f"Saved analysis CSV for folder {folder} to {summary_csv_path}")

if __name__ == "__main__":
    process_all_folders(base_dir, folders, output_dir)