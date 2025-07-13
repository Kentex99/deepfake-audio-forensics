import os
import pandas as pd
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Paths
base_output = r"E:\Research Project\output\Spectrograms"
comparison_output = r"E:\Research Project\output\Comparisons"
excel_file = r"E:\Research Project\matched_text_ids\matched_text_ids\matched_text_ids_unique.xlsx"

# Ensure output directories exist
for category in ["cloned", "anonymized", "synthetic"]:
    os.makedirs(os.path.join(comparison_output, category), exist_ok=True)

# Load matching information
matches = pd.read_excel(excel_file)

def load_image(path):
    """Load an image as grayscale."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def compare_images(img1, img2):
    """
    Compare two images using SSIM and MSE.

    Args:
        img1: First image (numpy array).
        img2: Second image (numpy array).

    Returns:
        ssim_val: Structural Similarity Index.
        mse_val: Mean Squared Error.
        diff: Difference image.
    """
    ssim_val, diff = ssim(img1, img2, full=True)
    mse_val = np.mean((img1 - img2) ** 2)
    diff = (diff * 255).astype(np.uint8)  # Scale difference to 0-255
    return ssim_val, mse_val, diff

def ensure_rgb(image):
    """
    Ensure that an image is 3-dimensional (RGB).

    Args:
        image: Input image (2D grayscale or 3D RGB).

    Returns:
        RGB image (3D).
    """
    if len(image.shape) == 2:  # Grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image  # Already RGB

def create_composite_image(human_colored, alt_colored, diff_colored, output_path, title):
    """
    Create and save a composite image showing the comparison.

    Args:
        human_colored: Human spectrogram (RGB).
        alt_colored: Alternative spectrogram (RGB).
        diff_colored: Difference image (RGB).
        output_path: Path to save the composite image.
        title: Title for the composite image.
    """
    # Resize to ensure all images have the same dimensions
    height = min(human_colored.shape[0], alt_colored.shape[0], diff_colored.shape[0])
    width = min(human_colored.shape[1], alt_colored.shape[1], diff_colored.shape[1])
    human_colored = cv2.resize(human_colored, (width, height))
    alt_colored = cv2.resize(alt_colored, (width, height))
    diff_colored = cv2.resize(diff_colored, (width, height))

    # Concatenate images horizontally
    composite = np.concatenate([human_colored, alt_colored, diff_colored], axis=1)

    # Save composite image
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display in matplotlib
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# Iterate through the matches
for _, row in matches.iterrows():
    human_file = os.path.join(base_output, "human", f"{row['human_file_name']}.png")
    for category in ["cloned", "anonymized", "synthetic"]:
        alt_file = os.path.join(base_output, category, f"{row[f'{category}_file_name']}.png")

        # Load images
        if os.path.exists(human_file) and os.path.exists(alt_file):
            human_img = load_image(human_file)
            alt_img = load_image(alt_file)

            # Compare images
            ssim_val, mse_val, diff_img = compare_images(human_img, alt_img)

            # Ensure all images are RGB
            human_colored = ensure_rgb(human_img)
            alt_colored = ensure_rgb(alt_img)
            diff_colored = ensure_rgb(cv2.applyColorMap(diff_img, cv2.COLORMAP_JET))

            # Create composite image
            output_image_path = os.path.join(comparison_output, category, f"{row['human_file_name']}_vs_{category}.png")
            create_composite_image(human_colored, alt_colored, diff_colored, output_image_path, title=f"Human vs {category.capitalize()}")

            # Save results to a CSV file
            result_csv = os.path.join(comparison_output, category, "comparison_results.csv")
            with open(result_csv, "a") as f:
                if os.path.getsize(result_csv) == 0:
                    f.write("human_file,alt_file,ssim,mse\n")  # Write header if file is empty
                f.write(f"{row['human_file_name']},{row[f'{category}_file_name']},{ssim_val},{mse_val}\n")

            print(f"Compared: {row['human_file_name']} vs {row[f'{category}_file_name']} ({category})")
