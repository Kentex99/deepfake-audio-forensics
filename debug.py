import os
import pandas as pd

# Excel file path
excel_file = r"E:\Research\matched_text_ids\matched_text_ids\matched_text_ids_unique.xlsx"

# Input directories
input_dirs = {
    "human": r"E:\Research\matched_text_ids\matched_text_ids\human",
    "cloned": r"E:\Research\matched_text_ids\matched_text_ids\cloned",
    "anonymized": r"E:\Research\matched_text_ids\matched_text_ids\anonymized",
    "synthetic": r"E:\Research\matched_text_ids\matched_text_ids\synthetic",
}

# Read Excel
df = pd.read_excel(excel_file)

# Test file existence for each category
categories = ["cloned", "anonymized", "synthetic"]

for category in categories:
    print(f"\nChecking {category} files...\n")
    
    for _, row in df.iterrows():
        human_file = row["human_file_name"]
        other_file = row[f"{category}_file_name"]

        # Construct file paths
        human_path = os.path.join(input_dirs["human"], human_file)
        other_path = os.path.join(input_dirs[category], other_file)

        # Check if files exist
        if not os.path.exists(human_path):
            print(f"❌ Human file not found: {human_path}")
        if not os.path.exists(other_path):
            print(f"❌ {category.capitalize()} file not found: {other_path}")
