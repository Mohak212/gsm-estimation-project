import os
import csv

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GSM_DIR = os.path.join(BASE_DIR, "data", "gsm")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "labels.csv")

# Image extensions to consider
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

rows = []

for folder_name in os.listdir(GSM_DIR):
    folder_path = os.path.join(GSM_DIR, folder_name)

    if not os.path.isdir(folder_path):
        continue

    # Folder format: serial-GSM (e.g., 1-173)
    try:
        gsm_value = int(folder_name.split("-")[1])
    except Exception:
        print(f"Skipping folder (invalid name): {folder_name}")
        continue

    for file_name in os.listdir(folder_path):
        ext = os.path.splitext(file_name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            image_path = os.path.join("gsm", folder_name, file_name)
            rows.append([image_path, gsm_value])

# Write CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "gsm"])
    writer.writerows(rows)

print(f"labels.csv created with {len(rows)} entries")
