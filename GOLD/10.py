"""
Filter GOLD Therapy ZIPs to keep only rows for cohort patients.
Single-run script (no SLURM needed).
"""
import os
import glob
import time
import zipfile
import pandas as pd

# ── Paths ──
FILTERED_CLINICAL_DIR = "/scratch/alice/b/bg205/DataCleaning_Gold_v2"
THERAPY_ZIPS = "/scratch/alice/b/bg205/DataCleaning_Gold_v2/GOLD/FZ_GOLD_All_Extract_Therapy_*.zip"
OUTPUT_DIR   = "/scratch/alice/b/bg205/28_02_GOLD/therapy"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_zip_txt(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        txt_members = [n for n in z.namelist() if n.lower().endswith(".txt")]
        if not txt_members:
            raise ValueError(f"No .txt in {zip_path}")
        with z.open(txt_members[0]) as f:
            return pd.read_csv(f, sep="\t", dtype=str, low_memory=False)


def load_cohort_patids():
    """Extract unique patids from filtered GOLD Clinical files."""
    pattern = os.path.join(FILTERED_CLINICAL_DIR, "Cleaned_GOLD_Extract_Clinical_*.txt")
    files = sorted(glob.glob(pattern))
    assert len(files) > 0, f"No filtered Clinical files found in {FILTERED_CLINICAL_DIR}"

    patids = set()
    for f in files:
        chunk = pd.read_csv(f, sep="\t", dtype=str, usecols=["patid"])
        patids.update(chunk["patid"].dropna().unique())
        print(f"  {os.path.basename(f)}: {len(chunk):,} rows, running total: {len(patids):,} patids")

    print(f"\nTotal cohort patids: {len(patids):,}\n")
    return patids


def main():
    start = time.perf_counter()

    # Step 1: Load cohort
    print(f"{'='*60}")
    print("Loading cohort patids from filtered Clinical files")
    print(f"{'='*60}\n")
    cohort_patids = load_cohort_patids()

    # Step 2: Filter each Therapy ZIP
    all_zips = sorted(glob.glob(THERAPY_ZIPS))
    print(f"{'='*60}")
    print(f"Filtering {len(all_zips)} Therapy ZIPs")
    print(f"{'='*60}\n")

    total_before = 0
    total_after = 0

    for i, zippath in enumerate(all_zips, start=1):
        t0 = time.perf_counter()
        print(f"----- {i}/{len(all_zips)}: {os.path.basename(zippath)} -----")

        df = read_zip_txt(zippath)
        before_rows = len(df)

        df = df[df["patid"].astype(str).isin(cohort_patids)]
        after_rows = len(df)

        total_before += before_rows
        total_after += after_rows

        print(f"  Rows: {before_rows:,} -> {after_rows:,}  (dropped {before_rows - after_rows:,})")
        print(f"  Patids kept: {df['patid'].nunique():,}")

        out_path = os.path.join(OUTPUT_DIR, f"Cleaned_GOLD_Therapy_{i}.txt")
        df.to_csv(out_path, sep="\t", index=False)
        print(f"  Saved to {out_path}  ({round(time.perf_counter() - t0, 1)}s)\n")

    print(f"{'='*60}")
    print(f"DONE — {len(all_zips)} files processed in {round((time.perf_counter() - start) / 60, 2)} min")
    print(f"Total rows: {total_before:,} -> {total_after:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
