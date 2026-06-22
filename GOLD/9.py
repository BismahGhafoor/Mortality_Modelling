"""
Extract unique patids from filtered GOLD Clinical files and save to a single file.
Run once before the Therapy filtering job.
"""
import os
import glob
import pandas as pd

FILTERED_CLINICAL_DIR = "/scratch/alice/b/bg205/DataCleaning_Gold_v2"
OUTPUT_PATH = "/scratch/alice/b/bg205/28_02_GOLD/therapy/gold_cohort_patids.txt"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

pattern = os.path.join(FILTERED_CLINICAL_DIR, "Cleaned_GOLD_Extract_Clinical_*.txt")
files = sorted(glob.glob(pattern))
assert len(files) > 0, f"No filtered Clinical files found in {FILTERED_CLINICAL_DIR}"

patids = set()
for f in files:
    chunk = pd.read_csv(f, sep="\t", dtype=str, usecols=["patid"])
    patids.update(chunk["patid"].dropna().unique())
    print(f"  {os.path.basename(f)}: {len(chunk):,} rows, running total: {len(patids):,} patids")

print(f"\nTotal unique patids: {len(patids):,}")

pd.DataFrame({"patid": sorted(patids)}).to_csv(OUTPUT_PATH, sep="\t", index=False)
print(f"Saved to {OUTPUT_PATH}")
