import os
import pandas as pd
import numpy as np
import librosa
import pyloudnorm as pyln  # Install with: pip install pyloudnorm
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import euclidean
import soundfile as sf

# 🔹 **Paths**
excel_path = r"E:\Research\matched_text_ids\matched_text_ids\matched_text_ids_unique.xlsx"
spectrogram_base_path = r"E:\Research\output\Spectrograms"
output_path = r"E:\Research\output\Comparison"

# 🔹 **Ensure output folders exist**
comparison_types = {
    "cloned_file_name": ("cloned", "human_vs_cloned"),
    "anonymized_file_name": ("anonymized", "human_vs_anonymized"),
    "synthetic_file_name": ("synthetic", "human_vs_synthetic")
}

for _, folder in comparison_types.values():
    os.makedirs(os.path.join(output_path, folder), exist_ok=True)

# 🔹 **Load Excel File**
df = pd.read_excel(excel_path)
df.columns = df.columns.str.strip().str.replace(r"\ufeff", "", regex=True)  # Remove BOM characters

# 🔹 **Ensure required columns exist**
expected_columns = ["human_file_name", "cloned_file_name", "anonymized_file_name", "synthetic_file_name"]
for col in expected_columns:
    if col not in df.columns:
        raise KeyError(f"Missing expected column: {col}. Found columns: {df.columns.tolist()}")

# 🔹 **Function to Find File with Any Extension**
def find_file_with_extension(folder, filename):
    """Searches for a file in a folder with any extension."""
    possible_extensions = [".png", ".jpg", ".wav"]  # Add other extensions if necessary
    for ext in possible_extensions:
        file_path = os.path.join(folder, filename + ext)
        if os.path.exists(file_path):
            return file_path
    return None  # File not found

# 🔹 **Define Processing Functions**
def compute_loudness_distance(ref_audio, test_audio, sr):
    meter = pyln.Meter(sr)
    loudness_ref = meter.integrated_loudness(ref_audio)
    loudness_test = meter.integrated_loudness(test_audio)
    return abs(loudness_ref - loudness_test)

def compute_buzzy_artifacts(audio, sr):
    lowcut = 4000  # Focus on high frequencies (>4 kHz)
    nyquist = 0.5 * sr
    low = lowcut / nyquist

    b, a = butter(6, low, btype='high')
    filtered_audio = filtfilt(b, a, audio)
    energy = np.sum(np.square(filtered_audio)) / len(filtered_audio)
    return energy

def process_audio_comparison(human_file, test_file, test_label, output_folder):
    """Process and compare human audio with a test file, calculating metrics."""
    if human_file is None:
        print(f"Skipping {test_label}: Missing human file")
        return None
    if test_file is None:
        print(f"Skipping {test_label}: Missing test file")
        return None

    # Load audio
    human_audio, sr = librosa.load(human_file, sr=None)
    test_audio, _ = librosa.load(test_file, sr=sr)

    # Compute metrics
    mse_value = np.mean((human_audio - test_audio) ** 2)
    euclidean_distance = euclidean(human_audio[:min(len(human_audio), len(test_audio))], 
                                   test_audio[:min(len(human_audio), len(test_audio))])
    loudness_diff = compute_loudness_distance(human_audio, test_audio, sr)
    buzzy_artifacts_human = compute_buzzy_artifacts(human_audio, sr)
    buzzy_artifacts_test = compute_buzzy_artifacts(test_audio, sr)

    # Save results
    result = {
        "Test Type": test_label,
        "MSE": mse_value,
        "Euclidean Distance": euclidean_distance,
        "Loudness Distance (ITU-R BS.1770)": loudness_diff,
        "Buzzy Artifacts (Human)": buzzy_artifacts_human,
        "Buzzy Artifacts (Test)": buzzy_artifacts_test
    }

    # Save the processed audio (optional)
    save_path = os.path.join(output_folder, os.path.basename(test_file))
    sf.write(save_path, test_audio, sr)

    return result

# 🔹 **Process Each Row in Excel File**
all_results = []
for index, row in df.iterrows():
    human_file = find_file_with_extension(os.path.join(spectrogram_base_path, "human"), row["human_file_name"])

    for test_col, (subfolder, folder) in comparison_types.items():
        test_file = find_file_with_extension(os.path.join(spectrogram_base_path, subfolder), row[test_col])
        result = process_audio_comparison(human_file, test_file, test_col, os.path.join(output_path, folder))
        if result:
            all_results.append(result)

# 🔹 **Save All Results to Excel**
output_excel = os.path.join(output_path, "comparison_results.xlsx")
results_df = pd.DataFrame(all_results)
results_df.to_excel(output_excel, index=False)
print(f"Processing complete. Results saved at {output_excel}")
