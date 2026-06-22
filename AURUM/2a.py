# -*- coding: utf-8 -*-
"""
Aurum Data Cleaning Script (reads .zip archives; writes chunks to filtered_aurum_chunks)
- Filters Observation (clinical) rows by medcode (and enttype if present)
- Optionally filters DrugIssue (therapy) rows by productcodeid
- Writes outputs to: /rfs/LRWE_Proj88/bg205/DataAnalysis/Data_Cleaning_AURUM/filtered_aurum_chunks
"""

import pandas as pd
import numpy as np
import time
import glob
import os
import zipfile
import warnings
import platform
import sys

warnings.simplefilter(action='ignore')

# =============================================================================
# DEBUG HELPERS (grep-friendly prefix "DBG|")
# =============================================================================
def dbg(df, name, id_col=None, date_cols=None):
    print(f"DBG| [{name}] rows={len(df):,}", end="")
    if id_col and id_col in df.columns:
        print(f"  {id_col}_unique={df[id_col].nunique():,}", end="")
    print()
    if date_cols:
        for c in date_cols:
            if c in df.columns:
                print(f"DBG|   - {c}: missing={df[c].isna().sum():,}")
                try:
                    yrs = pd.to_datetime(df[c], errors='coerce').dt.year
                    print(f"DBG|     year: min={yrs.min()} p50={yrs.median()} max={yrs.max()} "
                          f">2025={(yrs>2025).sum():,} <1900={(yrs<1900).sum():,}")
                except Exception as e:
                    print(f"DBG|     (year stats skipped: {e})")


def dbg_set_diff(before_set, after_set, label, print_max=20):
    lost = before_set - after_set
    gained = after_set - before_set
    print(f"DBG| [{label}] lost={len(lost):,}  gained={len(gained):,}")
    if lost:
        sample = sorted(lost)[:print_max]
        print(f"DBG|   lost sample (≤{print_max}): {sample}")


def dbg_date_raw_sample(df, col, n=5, label=""):
    """Show raw string values + dtype for a date column before any parsing."""
    if col not in df.columns:
        return
    print(f"DBG| [{label}] {col} dtype={df[col].dtype}  sample(≤{n}): {df[col].dropna().head(n).tolist()}")


# =============================================================================
# User Input
# =============================================================================
current_directory = '/scratch/alice/b/bg205/01_03_AURUM'
current_directory_hpc = '/scratch/alice/b/bg205/01_03_AURUM'

clinical_files_directory = "/scratch/alice/b/bg205/DataCleaning_Aurum_v2/input_observation/*.zip"
therapy_files_directory  = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/Aurum/DrugIssue/*.zip"

clinical_code_directory = "/scratch/alice/b/bg205/01_03_AURUM/filtered_diabetes_AURUM_codes.txt"
therapy_code_directory  = "final_codelist_gold_therapy.txt"

filter_clinical = True
filter_therapy  = False

OUTPUT_DIR = "/scratch/alice/b/bg205/01_03_AURUM/filtered_aurum_chunks"

# =============================================================================
# Functions
# =============================================================================
def change_directory(local_dir, hpc_dir=None):
    print("-" * 60)
    if platform.system() == 'Windows':
        path = local_dir
    elif platform.system() == 'Linux':
        path = hpc_dir or local_dir
    else:
        raise OSError("Unsupported operating system")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.chdir(path)
    print(f"Changed directory to: {os.getcwd()}")
    print(f"Output directory:     {OUTPUT_DIR}")
    print("-" * 60)

def read_codes(codelist_path, do_filter):
    if not do_filter:
        return None
    codes = pd.read_csv(codelist_path, sep='\t', dtype=str)

    # DBG: raw codelist inspection
    print(f"DBG| [READ_CODES] loaded {codelist_path}")
    print(f"DBG| [READ_CODES] columns={list(codes.columns)}  rows={len(codes):,}")

    colmap = {}
    if 'medcodeid' in codes.columns:
        colmap['medcodeid'] = 'code'
    if 'productcodeid' in codes.columns:
        colmap['productcodeid'] = 'code'
    if 'termtype' in codes.columns:
        colmap['termtype'] = 'terminology'
    if colmap:
        codes = codes.rename(columns=colmap)
    if 'terminology' not in codes.columns and 'code' in codes.columns:
        codes['terminology'] = 'medcode'
    assert len(codes) > 0, f'No codes found in {codelist_path}'

    pre_dropna = len(codes)
    codes = codes[['terminology','code']].dropna()
    post_dropna = len(codes)
    # DBG: dropna impact
    if pre_dropna != post_dropna:
        print(f"DBG| [READ_CODES] ⚠ dropna removed {pre_dropna - post_dropna:,} rows from codelist!")
    print(f"DBG| [READ_CODES] final codelist: {len(codes):,} rows, "
          f"unique codes={codes['code'].nunique():,}, "
          f"terminologies={codes['terminology'].value_counts().to_dict()}")
    # DBG: sample codes for visual sanity (whitespace, format)
    print(f"DBG| [READ_CODES] code sample (≤5): {codes['code'].head(5).tolist()}")
    print(f"DBG| [READ_CODES] code str lengths: min={codes['code'].str.len().min()} "
          f"max={codes['code'].str.len().max()}")

    return codes

def read_zip_all_txt(zip_path):
    frames = []
    skipped = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        members = [n for n in z.namelist() if n.lower().endswith(".txt")]
        print(f"DBG| [ZIP] {os.path.basename(zip_path)}: {len(members)} TXT member(s)")
        for name in members:
            with z.open(name) as f:
                try:
                    df = pd.read_csv(f, sep='\t', dtype=str, low_memory=False)
                    frames.append(df)
                except Exception as e:
                    skipped.append(name)
                    print(f"  Skipping {name} in {os.path.basename(zip_path)} (read error: {e})")
    if skipped:
        print(f"DBG| [ZIP] ⚠ skipped {len(skipped)} member(s): {skipped}")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

def read_files(files_glob):
    files = sorted(glob.glob(files_glob))
    assert len(files) > 0, f'No files found for pattern: {files_glob}'
    return files

# =============================================================================
# Clinical (Observation)
# =============================================================================
def append_clinical(files, do_filter=False, codes=None, task_id=0):
    start = time.perf_counter()
    medcodes = pd.Series(dtype=str)
    entities = pd.Series(dtype=str)
    if do_filter and codes is not None:
        medcodes = codes.loc[codes['terminology'].str.lower()=='medcode', 'code'].dropna().astype(str)
        if 'enttype' in codes['terminology'].str.lower().unique():
            entities = codes.loc[codes['terminology'].str.lower()=='enttype', 'code'].dropna().astype(str)

    # DBG: codelist that will be used for filtering
    print(f"DBG| [CLIN_FILTER_SETUP] medcodes to match: {len(medcodes):,}")
    if not entities.empty:
        print(f"DBG| [CLIN_FILTER_SETUP] enttypes to match: {len(entities):,}")
    else:
        print(f"DBG| [CLIN_FILTER_SETUP] no enttype filter")

    for idx, zipf in enumerate(files, start=1):
        df = read_zip_all_txt(zipf)
        print(f"[Clinical] {os.path.basename(zipf)} -> {len(df):,} rows before filter")

        if df.empty:
            print(f"DBG| [CLIN_EMPTY] ⚠ ZIP produced empty DataFrame, skipping.")
            continue

        # DBG: columns present + raw data shape
        print(f"DBG| [CLIN_RAW] columns={list(df.columns)}")
        dbg(df, "CLIN_RAW", id_col="patid")

        # DBG: date columns — early detection of parsing issues (obsdate is the key one)
        date_candidates = [c for c in df.columns if 'date' in c.lower()]
        if date_candidates:
            for dc in date_candidates:
                dbg_date_raw_sample(df, dc, n=5, label="CLIN_RAW_DATE")
                # Year sanity on raw strings
                try:
                    raw_yrs = pd.to_datetime(df[dc], errors='coerce').dt.year
                    n_nat = raw_yrs.isna().sum()
                    print(f"DBG| [CLIN_RAW_DATE] {dc}: NaT_after_coerce={n_nat:,}/{len(df):,} "
                          f"({100*n_nat/max(len(df),1):.1f}%)")
                    valid = raw_yrs.dropna()
                    if len(valid) > 0:
                        print(f"DBG| [CLIN_RAW_DATE] {dc} year: min={valid.min():.0f} "
                              f"p50={valid.median():.0f} max={valid.max():.0f} "
                              f">2025={(valid>2025).sum():,} <1900={(valid<1900).sum():,} "
                              f"==9999={(valid==9999).sum():,}")
                except Exception as e:
                    print(f"DBG| [CLIN_RAW_DATE] {dc} year stats skipped: {e}")

        # DBG: before filter patid checkpoint
        before_patids = set(df['patid'].unique()) if 'patid' in df.columns else set()
        before_rows = len(df)

        if do_filter and not df.empty:
            if "medcodeid" in df.columns:
                # DBG: check for format mismatches between codelist and data
                data_sample = df["medcodeid"].dropna().head(5).tolist()
                print(f"DBG| [CLIN_MEDCODE_FMT] data sample: {data_sample}")
                print(f"DBG| [CLIN_MEDCODE_FMT] data str lengths: "
                      f"min={df['medcodeid'].astype(str).str.len().min()} "
                      f"max={df['medcodeid'].astype(str).str.len().max()}")
                print(f"DBG| [CLIN_MEDCODE_FMT] codelist sample: {medcodes.head(5).tolist()}")

                if "enttype" in df.columns and not entities.empty:
                    df = df[
                        df["medcodeid"].astype(str).isin(medcodes) |
                        df["enttype"].astype(str).isin(entities)
                    ]
                else:
                    df = df[df["medcodeid"].astype(str).isin(medcodes)]
            else:
                print("  Warning: 'medcodeid' not in columns; no clinical filtering applied.")

        # DBG: after filter
        after_patids = set(df['patid'].unique()) if 'patid' in df.columns else set()
        print(f"DBG| [CLIN_FILTER] rows: {before_rows:,} → {len(df):,} "
              f"(dropped {before_rows - len(df):,}, {100*(before_rows - len(df))/max(before_rows,1):.1f}%)")
        dbg(df, "CLIN_FILTERED", id_col="patid")
        dbg_set_diff(before_patids, after_patids, "CLIN_FILTER_PATIDS")

        # DBG: date stats AFTER filter (to compare with before)
        if date_candidates and not df.empty:
            for dc in date_candidates:
                if dc in df.columns:
                    try:
                        post_yrs = pd.to_datetime(df[dc], errors='coerce').dt.year
                        n_nat = post_yrs.isna().sum()
                        print(f"DBG| [CLIN_FILT_DATE] {dc}: NaT={n_nat:,}/{len(df):,}")
                    except Exception:
                        pass

        # Write output
        out_path = os.path.join(OUTPUT_DIR, f"Cleaned_AURUM_Observation_{task_id}.txt")
        df.to_csv(out_path, sep='\t', index=False)
        print(f"  -> Wrote {len(df):,} rows to {out_path}")

    print(f"Completed in {round((time.perf_counter() - start)/60, 2)} mins")

# =============================================================================
# Therapy (DrugIssue)
# =============================================================================
def append_therapy(files, do_filter=False, codes=None):
    start = time.perf_counter()
    prodcodes = pd.Series(dtype=str)
    if do_filter and codes is not None:
        prodcodes = codes['code'].dropna().astype(str)

    # DBG: therapy codelist
    print(f"DBG| [THER_FILTER_SETUP] prodcodes to match: {len(prodcodes):,}")

    out_count = 0
    for idx, zipf in enumerate(files, start=1):
        df = read_zip_all_txt(zipf)
        print(f"[Therapy]  {idx}/{len(files)}: {os.path.basename(zipf)} -> {len(df):,} rows before filter")

        if df.empty:
            print(f"DBG| [THER_EMPTY] ⚠ ZIP produced empty DataFrame, skipping.")
            out_count += 1
            continue

        # DBG: raw shape
        dbg(df, "THER_RAW", id_col="patid")

        before_patids = set(df['patid'].unique()) if 'patid' in df.columns else set()
        before_rows = len(df)

        if do_filter and not df.empty:
            if "productcodeid" in df.columns:
                df = df[df["productcodeid"].astype(str).isin(prodcodes)]
            else:
                print("  Warning: 'productcodeid' not in columns; no therapy filtering applied.")

        # DBG: after filter
        after_patids = set(df['patid'].unique()) if 'patid' in df.columns else set()
        print(f"DBG| [THER_FILTER] rows: {before_rows:,} → {len(df):,} "
              f"(dropped {before_rows - len(df):,})")
        dbg(df, "THER_FILTERED", id_col="patid")
        dbg_set_diff(before_patids, after_patids, "THER_FILTER_PATIDS")

        out_count += 1
        out_path = os.path.join(OUTPUT_DIR, f"Cleaned_AURUM_DrugIssue_{out_count}.txt")
        df.to_csv(out_path, sep='\t', index=False)
        print(f"  -> Wrote {len(df):,} rows to {out_path}")

    print(f"All therapy files completed in {round((time.perf_counter() - start)/60, 2)} mins")

# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    change_directory(current_directory, current_directory_hpc)

    if len(sys.argv) < 2:
        print("ERROR: No task ID provided. Usage: python 2.py <SLURM_ARRAY_TASK_ID>")
        sys.exit(1)

    task_id = int(sys.argv[1])

    print("="*60 + f"\nProcessing Task {task_id}\n" + "="*60)

    clinical_codes = read_codes(clinical_code_directory, filter_clinical)
    all_clinical_files = sorted(glob.glob(clinical_files_directory))

    # DBG: file list summary
    print(f"DBG| [FILES] total clinical ZIPs found: {len(all_clinical_files)}")
    if all_clinical_files:
        print(f"DBG| [FILES] first: {all_clinical_files[0]}")
        print(f"DBG| [FILES] last:  {all_clinical_files[-1]}")

    if task_id >= len(all_clinical_files):
        print(f"Task ID {task_id} exceeds number of files ({len(all_clinical_files)}). Exiting.")
        sys.exit(0)

    single_file = [all_clinical_files[task_id]]
    print(f"This task will process: {single_file[0]}")

    append_clinical(single_file, filter_clinical, clinical_codes, task_id)
