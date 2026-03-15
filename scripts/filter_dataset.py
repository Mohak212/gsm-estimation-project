import csv
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_CSV = os.path.join(BASE_DIR, "data", "labels.csv")


def filter_by_gsm_range(min_gsm, max_gsm):
    """
    Returns list of (image_path, gsm) filtered by GSM range
    """
    filtered = []

    with open(LABELS_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gsm = int(row["gsm"])
            if min_gsm <= gsm <= max_gsm:
                filtered.append((row["image"], gsm))

    return filtered


if __name__ == "__main__":
    # TEST RUN (temporary)
    test_min = 90
    test_max = 110

    data = filter_by_gsm_range(test_min, test_max)
    print(f"Found {len(data)} images in range {test_min}-{test_max}")
