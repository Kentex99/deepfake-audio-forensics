import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Base input directory containing all categories
base_input_dir = r"E:\Research\matched_text_ids\matched_text_ids"

# Output base directory for spectrograms and vectors
output_base_dir = r"E:\Research\output"

# Categories to process
categories = ["human", "cloned", "anonymized", "synthetic"]

# Mel Spectrogram Settings
sr_target = 16000  # Target Sampling rate
n_mels = 128  # Number of Mel bands

def generate_mel_spectrogram(audio_path, output_spectrogram, output_vector):
    """
    Generate and save Mel Spectrograms in **Linear Scale** with enhanced contrast for better visualization.
    """
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=sr_target)

        # Debug: Check if the file loaded correctly
        if y is None or len(y) == 0:
            raise ValueError(f"Audio file {audio_path} is empty or could not be read.")

        # Generate Mel Spectrogram in Linear Scale (NO dB conversion)
        S_linear = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

        # Normalize spectrogram to 0-1 range for better visibility
        S_linear = S_linear / np.max(S_linear)

        # Apply contrast enhancement (raising to power 0.3 to make details clearer)
        S_enhanced = np.power(S_linear, 0.3)

        # Save Spectrogram Image (Enhanced Visualization)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_enhanced, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
        plt.colorbar(label="Amplitude (Enhanced Linear Scale)")
        plt.title("Mel Spectrogram (Linear Scale) - Enhanced")
        plt.savefig(output_spectrogram, bbox_inches="tight", dpi=300)
        plt.close()

        # Save Mel Spectrogram Vectors in Linear Scale (for Numerical Analysis)
        np.save(output_vector, S_linear)

        print(f"✅ Saved: {output_spectrogram} & {output_vector}")

    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")

def process_all_categories():
    """
    Process all categories (Human, Cloned, Anonymized, Synthetic) and generate Spectrograms & Vectors.
    """
    for category in categories:
        input_dir = os.path.join(base_input_dir, category)
        output_spectrogram_dir = os.path.join(output_base_dir, "Spectrograms", category)
        output_vector_dir = os.path.join(output_base_dir, "Vectors", category)

        # Ensure output directories exist
        os.makedirs(output_spectrogram_dir, exist_ok=True)
        os.makedirs(output_vector_dir, exist_ok=True)

        print(f"🔄 Processing {category} voice samples...")

        for file in os.listdir(input_dir):
            if file.endswith(".wav"):
                audio_path = os.path.join(input_dir, file)

                # Define output paths
                output_spectrogram = os.path.join(output_spectrogram_dir, f"{os.path.splitext(file)[0]}.png")
                output_vector = os.path.join(output_vector_dir, f"{os.path.splitext(file)[0]}.npy")

                # Generate Mel Spectrogram & Vector
                generate_mel_spectrogram(audio_path, output_spectrogram, output_vector)

if __name__ == "__main__":
    process_all_categories()
    print("✅ All Mel Spectrograms & Vectors generated successfully in Linear Scale.")