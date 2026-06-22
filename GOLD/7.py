#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SCRIPT 7 — Combine GOLD Test extracts into Test_entities_all.txt.gz (STREAMING)
"""

import pandas as pd
import glob
import os
import time

# =============================================================================
# Debug helpers
# =============================================================================
def dbg_date_check(series, name):
    """Quick date column stats on a chunk."""
    n_miss = series.isna().sum()
    n_total = len(series)
    info = f"DBG|[{name}] total={n_total:,}  missing={n_miss:,}"
    if pd.api.types.is_datetime64_any_dtype(series):
        valid = series.dropna()
        if len(valid) > 0:
            yrs = valid.dt.year
            info += (f"  year: min={yrs.min()} max={yrs.max()} "
                     f">2025={(yrs>2025).sum():,} <1900={(yrs<1900).sum():,}")
    else:
        sample = series.dropna().head(5).tolist()
        info += f"  raw sample: {sample}"
    print(info)


# =============================================================================
# Main
# =============================================================================
raw_zip_pattern = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Test_*.zip"
raw_txt_pattern = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Test_*.txt"

raw_test_files = sorted(glob.glob(raw_zip_pattern))
if not raw_test_files:
    raw_test_files = sorted(glob.glob(raw_txt_pattern))

print(f"Found {len(raw_test_files)} raw test files.")
if not raw_test_files:
    raise FileNotFoundError(
        f"No test files found for patterns:\n  {raw_zip_pattern}\n  {raw_txt_pattern}"
    )

output_filename = "Test_entities_all.txt.gz"
if os.path.exists(output_filename):
    os.remove(output_filename)
    print(f"Removed existing file: {output_filename}")

CHUNKSIZE = 20000

usecols = ["patid", "eventdate", "enttype", "data1", "data2", "data3"]

start = time.perf_counter()
written_any = False

# ── DBG: running counters ──
total_rows_read = 0
total_rows_written = 0
total_rows_dropped_date = 0
cumulative_patids = set()
per_file_patids_lost = {}
first_file_raw_sample_done = False

for idx, filename in enumerate(raw_test_files, start=1):
    print(f"\nProcessing file {idx}/{len(raw_test_files)}: {filename}")
    compression = "zip" if filename.lower().endswith(".zip") else "infer"

    reader = pd.read_csv(
        filename,
        sep="\t",
        dtype=str,
        chunksize=CHUNKSIZE,
        compression=compression,
        usecols=usecols,
    )

    file_rows_read = 0
    file_rows_written = 0
    file_rows_dropped = 0
    file_patids_before = set()
    file_patids_after = set()

    for chunk_num, chunk in enumerate(reader):
        file_rows_read += len(chunk)
        file_patids_before.update(chunk['patid'].astype(str).str.strip().unique())

        # ── DBG: raw eventdate sample (first chunk of first file only) ──
        if not first_file_raw_sample_done and chunk_num == 0:
            dbg_date_check(chunk["eventdate"], f"test_raw_eventdate_file{idx}")
            # also check enttype distribution on first chunk
            print(f"DBG|[test_file{idx}_chunk0] enttype value_counts (top 10):\n"
                  f"{chunk['enttype'].value_counts().head(10)}")
            first_file_raw_sample_done = True

        # Parse eventdate and drop invalid
        raw_missing = chunk["eventdate"].isna().sum() + (chunk["eventdate"].astype(str).isin(['', 'nan', 'None'])).sum()
        chunk["eventdate"] = pd.to_datetime(chunk["eventdate"], errors="coerce", dayfirst=True)
        parsed_missing = chunk["eventdate"].isna().sum()
        coerced = parsed_missing - raw_missing
        if coerced > 0 and chunk_num < 3:
            print(f"DBG|[test_file{idx}_chunk{chunk_num}] coerced_to_NaT={coerced:,}")

        rows_before_drop = len(chunk)
        chunk = chunk.dropna(subset=["eventdate"])
        dropped = rows_before_drop - len(chunk)
        file_rows_dropped += dropped

        if chunk.empty:
            continue

        file_patids_after.update(chunk['patid'].astype(str).str.strip().unique())

        # Keep raw test fields as-is
        chunk = chunk[["patid", "eventdate", "enttype", "data1", "data2", "data3"]]

        file_rows_written += len(chunk)

        # Write / append
        chunk.to_csv(
            output_filename,
            mode="a",
            header=not written_any,
            index=False,
            sep="\t",
            compression="gzip",
            date_format="%Y-%m-%d",
        )
        written_any = True

    # ── DBG: per-file summary ──
    total_rows_read += file_rows_read
    total_rows_written += file_rows_written
    total_rows_dropped_date += file_rows_dropped
    cumulative_patids.update(file_patids_after)

    pats_lost_this_file = file_patids_before - file_patids_after
    if pats_lost_this_file:
        per_file_patids_lost[idx] = len(pats_lost_this_file)

    elapsed = round((time.perf_counter() - start) / 60, 2)
    print(f"DBG|[test_file{idx}] rows_read={file_rows_read:,}  written={file_rows_written:,}  "
          f"dropped_date={file_rows_dropped:,}  "
          f"patids_in={len(file_patids_before):,}  patids_out={len(file_patids_after):,}  "
          f"patids_lost={len(pats_lost_this_file):,}")
    print(f"DBG|  cumulative patids={len(cumulative_patids):,}  (Elapsed: {elapsed} mins)")

# ── DBG: grand total ──
print(f"\n{'='*60}")
print(f"DBG|[test_TOTAL] rows_read={total_rows_read:,}  rows_written={total_rows_written:,}  "
      f"rows_dropped_by_date={total_rows_dropped_date:,}  "
      f"({total_rows_dropped_date/total_rows_read*100:.3f}%)" if total_rows_read > 0 else "")
print(f"DBG|[test_TOTAL] cumulative unique patids={len(cumulative_patids):,}")

if per_file_patids_lost:
    total_lost = sum(per_file_patids_lost.values())
    print(f"DBG|[test_TOTAL] files with patid loss: {len(per_file_patids_lost)}/{len(raw_test_files)}  "
          f"total patids lost across files (may overlap)={total_lost:,}")
    # NOTE: per-file patid loss is an overcount — a patient "lost" in file 1 may appear in file 2.
    # The true loss is patients in ANY raw file but not in any output — harder to compute in streaming.
else:
    print(f"DBG|[test_TOTAL] no patids lost to date filtering in any file")

# ── DBG: verify the output file can be read back ──
print(f"\nDBG|[test_verify_output] reading back {output_filename} ...")
try:
    df_verify = pd.read_csv(output_filename, sep="\t", compression="gzip", dtype=str, nrows=5)
    print(f"DBG|  columns={df_verify.columns.tolist()}")
    print(f"DBG|  sample:\n{df_verify.head()}")

    # Also check total row count (read just patid col for speed)
    df_count = pd.read_csv(output_filename, sep="\t", compression="gzip", usecols=["patid"], dtype=str)
    print(f"DBG|  total rows in file={len(df_count):,}  patids={df_count['patid'].nunique():,}")
    if len(df_count) != total_rows_written:
        print(f"DBG|  WARNING: row mismatch! written={total_rows_written:,} vs read_back={len(df_count):,}")
        print(f"DBG|  This may indicate gzip multi-stream truncation!")
    else:
        print(f"DBG|  Row count matches. Output file is intact.")
    del df_count
except Exception as e:
    print(f"DBG|  WARNING: failed to read back output: {e}")

print("\nTest_entities_all.txt.gz created successfully.")
