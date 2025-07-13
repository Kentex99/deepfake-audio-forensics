import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Directories
input_folder = r"E:\Research Project\cloned-original\027\cloned"  # Folder containing audio samples
output_images = r"E:\Research Project\Spectrograms\027\cloned\Spectrograms"  # Folder for spectrogram images
output_vectors = r"E:\Research Project\Spectrograms\027\cloned\Vectors"  # Folder for spectrogram vectors

# Ensure output directories exist
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_vectors, exist_ok=True)

# Function to save spectrograms
def save_spectrogram(file_path, image_path, vector_path, sr=22050):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=sr)

        # Compute the Short-Time Fourier Transform (STFT)
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Save spectrogram image
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()

        # Save vector representation
        np.save(vector_path, S_db)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Process all audio files in the input folder
file_list = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
total_files = len(file_list)

start_time = time.time()
for idx, file_name in enumerate(tqdm(file_list), start=1):
    file_path = os.path.join(input_folder, file_name)
    base_name = os.path.splitext(file_name)[0]

    # Paths for output files
    image_path = os.path.join(output_images, f"{base_name}.png")
    vector_path = os.path.join(output_vectors, f"{base_name}.npy")

    # Process and save
    save_spectrogram(file_path, image_path, vector_path)

    # Progress indicator with estimated time remaining
    elapsed_time = time.time() - start_time
    avg_time_per_file = elapsed_time / idx
    remaining_files = total_files - idx
    estimated_time_left = avg_time_per_file * remaining_files
    print(f"Processed {idx}/{total_files} files. Estimated time left: {estimated_time_left:.2f} seconds.", end="\r")

print("\nProcessing complete.")

