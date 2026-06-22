# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
import glob
import os
import warnings
import platform
import zipfile

warnings.simplefilter(action='ignore')


# =============================================================================
# Debug helpers
# =============================================================================
def dbg(df, name, date_cols=None):
    pid_info = f"  patids={df['patid'].nunique():,}" if 'patid' in df.columns else ""
    print(f"DBG|[{name}] rows={len(df):,}{pid_info}")
    if date_cols:
        for c in date_cols:
            if c not in df.columns:
                continue
            n_miss = df[c].isna().sum()
            print(f"DBG|  - {c}: missing={n_miss:,}  dtype={df[c].dtype}")
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                valid = df[c].dropna()
                if len(valid) > 0:
                    yrs = valid.dt.year
                    print(f"DBG|    year(dt): min={yrs.min()} p50={yrs.median()} max={yrs.max()} "
                          f">2025={(yrs>2025).sum():,} <1900={(yrs<1900).sum():,}")
            else:
                sample = df.loc[df[c].notna(), c].head(5).tolist()
                print(f"DBG|    raw sample: {sample}")
                try:
                    raw_str = df[c].astype(str)
                    yr = pd.to_numeric(raw_str.str[-4:], errors='coerce')
                    print(f"DBG|    year(str): min={yr.min()} p50={yr.median()} max={yr.max()} "
                          f">2025={(yr>2025).sum():,} <1900={(yr<1900).sum():,} ==9999={(yr==9999).sum():,}")
                except Exception as e:
                    print(f"DBG|    (year str check skipped: {e})")


def dbg_patid_diff(before_patids, after_patids, step_name):
    lost = before_patids - after_patids
    gained = after_patids - before_patids
    print(f"DBG|[{step_name}] patids before={len(before_patids):,}  after={len(after_patids):,}  "
          f"lost={len(lost):,}  gained={len(gained):,}")
    if lost:
        print(f"DBG|  lost (first 20): {sorted(lost)[:20]}")


# =============================================================================
# User Input and Configuration
# =============================================================================
current_directory = '/scratch/alice/b/bg205/28_02_GOLD'
current_directory_hpc = '/scratch/alice/b/bg205/28_02_GOLD'

clinical_zip_pattern = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Clinical_*.zip"
clinical_txt_pattern = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Clinical_*.txt"

smoking_codes_excel = "/scratch/alice/b/bg205/DataCleaning_FINAL_Gold/GOLD_Codes_FZ.xlsx"
smoking_sheet_name = "Smok"

max_rows_limit = 20000  # unused here (kept to match your style)
final_columns = ["patid", "eventdate", "medcode"]

# =============================================================================
# Helpers
# =============================================================================
def change_directory(current_directory, current_directory_hpc=None):
    print(f"{'-'*60}")
    path = current_directory if platform.system() == 'Windows' else current_directory_hpc
    os.chdir(path)
    print(f"Changed directory to: {os.getcwd()}")
    print(f"{'-'*60}\n")

def read_files():
    """Return a sorted list of clinical files (.zip preferred, else .txt)."""
    files = sorted(glob.glob(clinical_zip_pattern))
    if not files:
        files = sorted(glob.glob(clinical_txt_pattern))
    assert files, f"No clinical files found for patterns:\n  {clinical_zip_pattern}\n  {clinical_txt_pattern}"
    exts = {os.path.splitext(f)[1].lower() for f in files}
    print(f"\nFound {len(files)} clinical files with extension(s): {exts}\n")
    return files

def read_tab_maybe_zip(path, usecols=None):
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            if not members:
                raise FileNotFoundError(f"No .txt found inside {path}")
            inner = members[0]
            with zf.open(inner) as fh:
                df = pd.read_csv(fh, sep="\t", header=0, dtype=str, usecols=usecols, low_memory=False)
            print(f"  -> read {inner}")
            return df
    else:
        return pd.read_csv(path, sep="\t", header=0, dtype=str, usecols=usecols, low_memory=False)

# =============================================================================
# Smoking Extraction (Clinical Only)
# =============================================================================
def append_clinical_smoking(files, smoking_medcodes):
    start = time.perf_counter()
    print("\nStarting processing of clinical files for smoking medcodes...\n")

    usecols = ["patid", "eventdate", "medcode"]
    parts = []
    cumulative_patids = set()
    total_rows_dropped_date = 0
    total_pats_lost_date = set()

    for idx, filename in enumerate(files, start=1):
        print(f'Processing file {idx}/{len(files)}: {os.path.basename(filename)}')
        df = read_tab_maybe_zip(filename, usecols=usecols)
        print(f'  Total rows read: {len(df):,}')

        # Filter FIRST (faster)
        df["medcode"] = df["medcode"].astype(str).str.strip()
        df_filtered = df[df["medcode"].isin(smoking_medcodes)].copy()
        print(f'  Rows after filtering for smoking medcodes: {len(df_filtered):,}')

        if df_filtered.empty:
            elapsed = round((time.perf_counter() - start) / 60, 2)
            print(f'  No rows kept (Elapsed: {elapsed} mins)\n')
            continue

        # ── DBG: raw eventdate before parsing (first 3 files) ──
        if idx <= 3:
            dbg(df_filtered, f"smoking_pre_parse_file{idx}", date_cols=["eventdate"])

        # ── DBG: checkpoint patids before date parse + dropna ──
        pats_before_date = set(df_filtered['patid'].astype(str).str.strip().unique())

        # Parse dates only for filtered rows
        df_filtered["eventdate"] = pd.to_datetime(df_filtered["eventdate"], errors="coerce", dayfirst=True)

        # ── DBG: how many became NaT? ──
        n_nat_after_parse = df_filtered["eventdate"].isna().sum()

        rows_before_dropna = len(df_filtered)
        df_filtered = df_filtered.dropna(subset=["eventdate"])
        rows_dropped = rows_before_dropna - len(df_filtered)
        total_rows_dropped_date += rows_dropped

        # ── DBG: patid diff after date dropna ──
        pats_after_date = set(df_filtered['patid'].astype(str).str.strip().unique()) if len(df_filtered) > 0 else set()
        pats_lost_this_file = pats_before_date - pats_after_date
        total_pats_lost_date.update(pats_lost_this_file)

        if idx <= 3 or rows_dropped > 0:
            print(f"DBG|[smoking_date_filter_file{idx}] NaT_after_parse={n_nat_after_parse:,}  "
                  f"rows_dropped_by_dropna={rows_dropped:,}  "
                  f"patids_lost={len(pats_lost_this_file):,}")
            if pats_lost_this_file and len(pats_lost_this_file) <= 20:
                print(f"DBG|  lost patids: {sorted(pats_lost_this_file)}")

        # Clean patid type
        df_filtered["patid"] = df_filtered["patid"].astype(str).str.strip()

        # Keep only required output columns
        df_filtered = df_filtered[final_columns]

        parts.append(df_filtered)
        cumulative_patids.update(df_filtered['patid'].unique())

        elapsed = round((time.perf_counter() - start) / 60, 2)
        print(f'  Accumulated chunks: {len(parts)}  cumulative patids: {len(cumulative_patids):,}  '
              f'(Elapsed: {elapsed} mins)\n')

    if parts:
        final_df = pd.concat(parts, ignore_index=True)
    else:
        final_df = pd.DataFrame(columns=final_columns)

    memory_size = np.round(final_df.memory_usage(deep=True).sum() / (1024**2), 1)

    # ── DBG: final summary ──
    print(f"\n{'='*60}")
    print(f"DBG|[smoking_FINAL] rows={len(final_df):,}  patids={final_df['patid'].nunique():,}")
    print(f"DBG|[smoking_FINAL] total_rows_dropped_by_date_dropna={total_rows_dropped_date:,}")
    print(f"DBG|[smoking_FINAL] total_patids_lost_entirely_due_to_date={len(total_pats_lost_date):,}")
    if total_pats_lost_date:
        print(f"DBG|  (first 20): {sorted(total_pats_lost_date)[:20]}")
    dbg(final_df, "smoking_final_df", date_cols=["eventdate"])

    # ── DBG: year distribution in final output ──
    if len(final_df) > 0 and pd.api.types.is_datetime64_any_dtype(final_df["eventdate"]):
        yrs = final_df["eventdate"].dt.year
        print(f"DBG|[smoking_final_years] min={yrs.min()} p50={yrs.median()} max={yrs.max()} "
              f">2025={(yrs>2025).sum():,} <1900={(yrs<1900).sum():,}")

    print(f"\nFinal smoking clinical dataset: {len(final_df):,} rows, Memory usage: {memory_size} MB")

    output_filename = "Clinical_SmokingStatus_all.txt.gz"
    final_df.to_csv(output_filename, sep="\t", index=False, compression="gzip", date_format='%d/%m/%Y')
    finish = time.perf_counter()
    print(f"File '{output_filename}' created in {round((finish - start)/60, 2)} mins.\n")

# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    change_directory(current_directory, current_directory_hpc)

    smoking_codes_df = pd.read_excel(smoking_codes_excel, sheet_name=smoking_sheet_name, dtype=str)
    smoking_medcodes = smoking_codes_df['medcode'].dropna().astype(str).str.strip().unique().tolist()
    print(f"DBG|[smoking_codelist] Total smoking medcodes loaded: {len(smoking_medcodes):,}")
    print(f"DBG|[smoking_codelist] sample (first 10): {smoking_medcodes[:10]}")

    clinical_files = read_files()
    append_clinical_smoking(clinical_files, set(smoking_medcodes))
