# -*- coding: utf-8 -*-
"""
Append CPRD GOLD extracts from ZIP archives (each containing a .txt),
optionally filter by code lists, and write chunked outputs.

FIXES APPLIED (minimal + pipeline-preserving):
1) Clinical diabetes-filter output:
   - Filter ONLY by medcode codelist (Script 1 produces terminology='medcode' only).
   - Removes the dead enttype-codelist branch that was always empty.
2) Adds a second Clinical output for Additional-date recovery:
   - Writes slim KEY table from FULL Clinical (no filtering):
     patid, adid, enttype, eventdate
   - This is what you merge onto Additional on (patid, adid, enttype).
3) Makes code loading robust to code column being either 'code' or 'medcode'.
"""

import pandas as pd
import numpy as np
import time
import glob
import os
import warnings
import platform
import zipfile

warnings.simplefilter(action="ignore")


# =============================================================================
# Debug helper
# =============================================================================
def dbg(df, name, date_cols=None):
    print(f"DBG|[{name}] rows={len(df):,}  patids={df['patid'].nunique():,}" if 'patid' in df.columns
          else f"DBG|[{name}] rows={len(df):,}")
    if date_cols:
        for c in date_cols:
            if c not in df.columns:
                continue
            missing = df[c].isna().sum() + (df[c].astype(str).isin(['', 'nan', 'NaT', 'None'])).sum()
            print(f"DBG|  - {c}: missing/blank={missing:,}  dtype={df[c].dtype}")
            sample = df.loc[df[c].notna() & ~df[c].astype(str).isin(['', 'nan']), c].head(5).tolist()
            print(f"DBG|    raw sample: {sample}")
            try:
                raw_str = df[c].astype(str)
                yr_last4 = pd.to_numeric(raw_str.str[-4:], errors='coerce')
                yr_first4 = pd.to_numeric(raw_str.str[:4], errors='coerce')
                yr = yr_last4 if yr_last4.between(1800, 2100).sum() > yr_first4.between(1800, 2100).sum() else yr_first4
                print(f"DBG|    year(str): min={yr.min()} p50={yr.median()} max={yr.max()} "
                      f">2025={(yr > 2025).sum():,} <1900={(yr < 1900).sum():,} ==9999={(yr == 9999).sum():,}")
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
# User Input
# =============================================================================
current_directory       = "/scratch/alice/b/bg205/28_02_GOLD"
current_directory_hpc   = "/scratch/alice/b/bg205/28_02_GOLD"

clinical_files_directory = "/scratch/alice/b/bg205/DataCleaning_Gold_v2/GOLD/FZ_GOLD_All_Extract_Clinical_*.zip"
therapy_files_directory  = "/scratch/alice/b/bg205/DataCleaning_Gold_v2/GOLD/FZ_GOLD_All_Extract_Therapy_*.zip"
test_files_directory     = "/scratch/alice/b/bg205/DataCleaning_Gold_v2/GOLD/FZ_GOLD_All_Extract_Test_*.zip"

clinical_code_directory  = "/scratch/alice/b/bg205/28_02_GOLD/filtered_diabetes_codes.txt"
therapy_code_directory   = "final_codelist_gold_therapy.txt"
test_code_directory      = "final_codelist_gold_test.txt"

filter_clinical = True
filter_therapy  = False
filter_test     = False

max_rows_limit  = 2_000_000


# =============================================================================
# Helpers
# =============================================================================
def change_directory(current_directory, current_directory_hpc=None):
    print(f"{'-'*60}")
    if platform.system() == "Windows":
        path = current_directory
    elif platform.system() == "Linux":
        path = current_directory_hpc
    else:
        raise OSError("Unsupported operating system")

    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    print(f"Changed directory to: {os.getcwd()}")
    print(f"{'-'*60}\n")


def read_zip_txt_file(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        txt_members = [name for name in z.namelist() if name.lower().endswith(".txt")]
        if not txt_members:
            raise ValueError(f"No .txt files found in archive: {zip_path}")
        if len(txt_members) > 1:
            print(f"Warning: multiple .txt in {os.path.basename(zip_path)}; reading '{txt_members[0]}'")

        with z.open(txt_members[0]) as f:
            return pd.read_csv(f, sep="\t", dtype=str, low_memory=False)


def read_files(files_glob):
    files = sorted(glob.glob(files_glob))
    if not files:
        raise FileNotFoundError(f"No files found in: {files_glob}")

    exts = {os.path.splitext(f)[1].lower() for f in files}
    print(f"{'-'*40}")
    if len(exts) > 1:
        raise ValueError(f"Mixed file types in {files_glob}: {exts}")
    print(f"Found {len(files)} files. Extension: {list(exts)[0]}")
    print(f"{'-'*40}\n")
    return files


def read_codes(codelist_path, do_filter):
    """
    Supports codes files where the code column is either 'code' or 'medcode'.
    Requires a 'terminology' column (your Script 1 sets terminology='medcode').
    """
    if not do_filter:
        return None

    codes = pd.read_csv(codelist_path, sep="\t", dtype=str, low_memory=False)
    if len(codes) == 0:
        raise ValueError(f"No codes found in: {codelist_path}")

    if "terminology" not in codes.columns:
        raise ValueError(f"Expected 'terminology' column in {codelist_path}")

    code_col = "code" if "code" in codes.columns else ("medcode" if "medcode" in codes.columns else None)
    if code_col is None:
        raise ValueError(f"Expected 'code' or 'medcode' column in {codelist_path}")

    codes = codes[["terminology", code_col]].dropna().copy()

    # Normalise to a 'code' column name for downstream use
    if code_col != "code":
        codes = codes.rename(columns={code_col: "code"})

    print("Total codes found:")
    for term, count in codes["terminology"].value_counts(dropna=False).items():
        print(f"  {term}: {count}")
    print(f"{'-'*40}\n")
    return codes


def ensure_columns(df, required, context=""):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {context}. Available: {list(df.columns)[:20]}...")


def flush_chunk(dfs, out_prefix, chunk_number):
    if not dfs:
        return chunk_number, []

    out_df = pd.concat(dfs, ignore_index=True)
    
    if out_prefix == "Clinical_KEYS_for_Additional":
    out_df = out_df.drop_duplicates(subset=["patid", "adid", "enttype"], keep="last")
    
    size_mb = out_df.memory_usage(deep=True).sum() / (1024**2)

    out_path = f"{out_prefix}_{chunk_number}.txt"
    print("-" * 40)
    print(f"Saving {out_path} ({len(out_df):,} rows, {size_mb:.1f} MB) ...")

    dbg(out_df, f"flush_{out_prefix}_chunk{chunk_number}",
        date_cols=[c for c in out_df.columns if 'date' in c.lower()])

    out_df.to_csv(out_path, sep="\t", index=False)
    print("Saved.\n")

    return chunk_number + 1, []


# =============================================================================
# Append functions (RAM-safe)
# =============================================================================
def append_clinical(zip_files, do_filter=False, codes=None, max_rows_limit=np.inf):
    """
    Output 1: diabetes-filtered Clinical for baseline/indexdate work (Script 3).
    Filter is MEDCODE-ONLY (because your Script 1 creates terminology='medcode' only).
    """
    out_prefix = "Cleaned_GOLD_Extract_Clinical"
    chunk_number = 1
    dfs = []
    rows_in_buffer = 0
    start = time.perf_counter()
    cumulative_patids = set()

    medcodes = None
    if do_filter:
        medcodes = set(codes.loc[codes["terminology"] == "medcode", "code"].astype(str))
        print(f"DBG|[clinical_codelist] medcodes={len(medcodes):,}")

    for i, zippath in enumerate(zip_files, start=1):
        print(f"----- Clinical {i}/{len(zip_files)}: {os.path.basename(zippath)} -----")
        df = read_zip_txt_file(zippath)
        print(f"Rows read: {len(df):,}")

        ensure_columns(df, ["patid", "eventdate"], context=f"Clinical file {os.path.basename(zippath)}")

        if i <= 3:
            dbg(df, f"clinical_raw_file{i}", date_cols=["eventdate"])

        if do_filter:
            ensure_columns(df, ["medcode"], context=f"Clinical file {os.path.basename(zippath)}")
            df["medcode"] = df["medcode"].astype(str)

            before_pats = set(df["patid"].unique())
            df = df[df["medcode"].isin(medcodes)]
            after_pats = set(df["patid"].unique())
            dbg_patid_diff(before_pats, after_pats, f"clinical_medcode_filter_file{i}")
            print(f"Rows after filter: {len(df):,}")

        if len(df) > 0:
            dfs.append(df)
            rows_in_buffer += len(df)
            cumulative_patids.update(df["patid"].unique())

        print(f"Buffered rows: {rows_in_buffer:,}  cumulative patids so far: {len(cumulative_patids):,}")
        print(f"Elapsed: {round((time.perf_counter()-start)/60, 2)} min\n")

        if rows_in_buffer >= max_rows_limit:
            chunk_number, dfs = flush_chunk(dfs, out_prefix, chunk_number)
            rows_in_buffer = 0

    if rows_in_buffer > 0:
        chunk_number, dfs = flush_chunk(dfs, out_prefix, chunk_number)

    print(f"DBG|[clinical_TOTAL] cumulative unique patids={len(cumulative_patids):,}")
    print(f"All Clinical done in {round((time.perf_counter()-start)/60, 2)} min")
    print(f"{'-'*60}\n")


def append_clinical_keys(zip_files, max_rows_limit=np.inf):
    """
    Output 2: FULL Clinical keys for Additional-date recovery (Sharmin check).
    Keeps only: patid, adid, enttype, eventdate
    No diabetes filtering.
    """
    out_prefix = "Clinical_KEYS_for_Additional"
    chunk_number = 1
    dfs = []
    rows_in_buffer = 0
    start = time.perf_counter()
    cumulative_patids = set()

    for i, zippath in enumerate(zip_files, start=1):
        print(f"----- Clinical-KEYS {i}/{len(zip_files)}: {os.path.basename(zippath)} -----")
        df = read_zip_txt_file(zippath)
        print(f"Rows read: {len(df):,}")

        ensure_columns(df, ["patid", "adid", "enttype", "eventdate"],
                       context=f"Clinical KEYS file {os.path.basename(zippath)}")

        df = df[["patid", "adid", "enttype", "eventdate"]].copy()

        for c in ["patid", "adid", "enttype", "eventdate"]:
            df[c] = df[c].astype(str).str.strip()
        
        df.loc[df["eventdate"].isin(["", "nan", "None", "NaT"]), "eventdate"] = np.nan

        # Deduplicate to avoid row explosion when merging Additional->Clinical
        df = df.drop_duplicates(subset=["patid", "adid", "enttype"], keep="last")

        if i <= 3:
            dbg(df, f"clinical_keys_raw_file{i}", date_cols=["eventdate"])

        if len(df) > 0:
            dfs.append(df)
            rows_in_buffer += len(df)
            cumulative_patids.update(df["patid"].unique())

        print(f"Buffered rows: {rows_in_buffer:,}  cumulative patids so far: {len(cumulative_patids):,}")
        print(f"Elapsed: {round((time.perf_counter()-start)/60, 2)} min\n")

        if rows_in_buffer >= max_rows_limit:
            chunk_number, dfs = flush_chunk(dfs, out_prefix, chunk_number)
            rows_in_buffer = 0

    if rows_in_buffer > 0:
        chunk_number, dfs = flush_chunk(dfs, out_prefix, chunk_number)

    print(f"DBG|[clinical_KEYS_TOTAL] cumulative unique patids={len(cumulative_patids):,}")
    print(f"All Clinical-KEYS done in {round((time.perf_counter()-start)/60, 2)} min")
    print(f"{'-'*60}\n")


def append_therapy(zip_files, do_filter=False, codes=None, max_rows_limit=np.inf):
    out_prefix = "Cleaned_GOLD_Extract_Therapy"
    chunk_number = 1
    dfs = []
    rows_in_buffer = 0
    start = time.perf_counter()
    cumulative_patids = set()

    if do_filter:
        prodcodes = set(codes.loc[codes["terminology"] == "prodcode", "code"].astype(str)) \
            if "terminology" in codes.columns else set(codes["code"].astype(str))
        print(f"DBG|[therapy_codelist] prodcodes={len(prodcodes):,}")

    for i, zippath in enumerate(zip_files, start=1):
        print(f"----- Therapy {i}/{len(zip_files)}: {os.path.basename(zippath)} -----")
        df = read_zip_txt_file(zippath)
        print(f"Rows read: {len(df):,}")

        ensure_columns(df, ["patid"], context=f"Therapy file {os.path.basename(zippath)}")

        if i <= 3:
            date_cols_t = [c for c in df.columns if 'date' in c.lower()]
            if date_cols_t:
                dbg(df, f"therapy_raw_file{i}", date_cols=date_cols_t)

        if do_filter:
            ensure_columns(df, ["prodcode"], context=f"Therapy file {os.path.basename(zippath)}")
            before_pats = set(df['patid'].unique())
            df = df[df["prodcode"].isin(prodcodes)]
            after_pats = set(df['patid'].unique())
            dbg_patid_diff(before_pats, after_pats, f"therapy_code_filter_file{i}")
            print(f"Rows after filter: {len(df):,}")

        if len(df) > 0:
            dfs.append(df)
            rows_in_buffer += len(df)
            cumulative_patids.update(df['patid'].unique())

        print(f"Buffered rows: {rows_in_buffer:,}  cumulative patids so far: {len(cumulative_patids):,}")
        print(f"Elapsed: {round((time.perf_counter()-start)/60, 2)} min\n")

        if rows_in_buffer >= max_rows_limit:
            chunk_number, dfs = flush_chunk(dfs, out_prefix, chunk_number)
            rows_in_buffer = 0

    if rows_in_buffer > 0:
        chunk_number, dfs = flush_chunk(dfs, out_prefix, chunk_number)

    print(f"DBG|[therapy_TOTAL] cumulative unique patids={len(cumulative_patids):,}")
    print(f"All Therapy done in {round((time.perf_counter()-start)/60, 2)} min")
    print(f"{'-'*60}\n")


def append_test(zip_files, do_filter=False, codes=None, max_rows_limit=np.inf):
    out_prefix = "Cleaned_GOLD_Extract_Test"
    chunk_number = 1
    dfs = []
    rows_in_buffer = 0
    start = time.perf_counter()
    cumulative_patids = set()

    if do_filter:
        entities = set(codes.loc[codes["terminology"] == "enttype", "code"].astype(str)) \
            if "terminology" in codes.columns else set(codes["code"].astype(str))
        print(f"DBG|[test_codelist] enttypes={len(entities):,}")

    for i, zippath in enumerate(zip_files, start=1):
        print(f"----- Test {i}/{len(zip_files)}: {os.path.basename(zippath)} -----")
        df = read_zip_txt_file(zippath)
        print(f"Rows read: {len(df):,}")

        ensure_columns(df, ["patid"], context=f"Test file {os.path.basename(zippath)}")

        if i <= 3:
            date_cols_te = [c for c in df.columns if 'date' in c.lower()]
            if date_cols_te:
                dbg(df, f"test_raw_file{i}", date_cols=date_cols_te)

        if do_filter:
            ensure_columns(df, ["enttype"], context=f"Test file {os.path.basename(zippath)}")
            before_pats = set(df['patid'].unique())
            df = df[df["enttype"].isin(entities)]
            after_pats = set(df['patid'].unique())
            dbg_patid_diff(before_pats, after_pats, f"test_code_filter_file{i}")
            print(f"Rows after filter: {len(df):,}")

        if len(df) > 0:
            dfs.append(df)
            rows_in_buffer += len(df)
            cumulative_patids.update(df['patid'].unique())

        print(f"Buffered rows: {rows_in_buffer:,}  cumulative patids so far: {len(cumulative_patids):,}")
        print(f"Elapsed: {round((time.perf_counter()-start)/60, 2)} min\n")

        if rows_in_buffer >= max_rows_limit:
            chunk_number, dfs = flush_chunk(dfs, out_prefix, chunk_number)
            rows_in_buffer = 0

    if rows_in_buffer > 0:
        chunk_number, dfs = flush_chunk(dfs, out_prefix, chunk_number)

    print(f"DBG|[test_TOTAL] cumulative unique patids={len(cumulative_patids):,}")
    print(f"All Test done in {round((time.perf_counter()-start)/60, 2)} min")
    print(f"{'-'*60}\n")


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    change_directory(current_directory, current_directory_hpc)

    print(f"{'='*60}\nAppending Clinical Files (DIABETES-FILTERED for baseline/indexdate)\n{'='*60}\n")
    clinical_codes = read_codes(clinical_code_directory, filter_clinical)
    clinical_files = read_files(clinical_files_directory)
    append_clinical(clinical_files, filter_clinical, clinical_codes, max_rows_limit=max_rows_limit)

    print(f"{'='*60}\nBuilding Clinical KEYS (FULL clinical) for Additional-date recovery\n{'='*60}\n")
    append_clinical_keys(clinical_files, max_rows_limit=max_rows_limit)

    print(f"{'='*60}\nAppending Therapy Files\n{'='*60}\n")
    therapy_codes = read_codes(therapy_code_directory, filter_therapy)
    therapy_files = read_files(therapy_files_directory)
    append_therapy(therapy_files, filter_therapy, therapy_codes, max_rows_limit=max_rows_limit)

    print(f"{'='*60}\nAppending Test Files\n{'='*60}\n")
    test_codes = read_codes(test_code_directory, filter_test)
    test_files = read_files(test_files_directory)
    append_test(test_files, filter_test, test_codes, max_rows_limit=max_rows_limit)
