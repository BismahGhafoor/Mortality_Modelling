#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SCRIPT 3 (CORRECTED): Build GOLD baseline cohort with INDEXDATE
  – Now matches AURUM logic: keep all records → drop NA dates → map types
    → filter type 1 & 2 → THEN earliest per patient.
"""

import os
import glob
import pandas as pd
import numpy as np


# =============================================================================
# Debug helpers
# =============================================================================
def dbg(df, name, id_col=None, date_cols=None):
    print(f"DBG| [{name}] rows={len(df):,}", end="")
    if id_col and id_col in df.columns:
        print(f"  patids={df[id_col].nunique():,}", end="")
    print()
    if date_cols:
        for c in date_cols:
            if c in df.columns:
                print(f"DBG|   - {c}: missing={df[c].isna().sum():,}")
                try:
                    if pd.api.types.is_datetime64_any_dtype(df[c]):
                        yrs = df[c].dt.year
                    else:
                        yrs = pd.to_datetime(df[c], errors='coerce').dt.year
                    valid = yrs.dropna()
                    if len(valid) > 0:
                        print(f"DBG|     year: min={valid.min():.0f} p50={valid.median():.0f} "
                              f"max={valid.max():.0f} >2025={(valid>2025).sum():,} "
                              f"<1900={(valid<1900).sum():,} ==9999={(valid==9999).sum():,}")
                except Exception as e:
                    print(f"DBG|     (year stats skipped: {e})")


def dbg_set_diff(before_set, after_set, label, print_max=20):
    lost = before_set - after_set
    gained = after_set - before_set
    print(f"DBG| [{label}] lost={len(lost):,}  gained={len(gained):,}")
    if lost:
        sample = sorted(lost)[:print_max]
        print(f"DBG|   lost sample (≤{print_max}): {sample}")


# =============================================================================
# Paths
# =============================================================================
chunk_dir = "/scratch/alice/b/bg205/28_02_GOLD"
chunk_glob = os.path.join(chunk_dir, "Cleaned_GOLD_Extract_Clinical_*.txt")

codes_file = os.path.join(chunk_dir, "filtered_diabetes_codes.txt")

out_baseline_ungrouped = os.path.join(chunk_dir, "gold_baseline_ungrouped_df_NoNA.txt")
out_baseline_grouped   = os.path.join(chunk_dir, "gold_baseline_grouped_df_NoNA.txt")
out_type1              = os.path.join(chunk_dir, "gold_baseline_Type_1_Diabetes_NoNA.txt")
out_type2              = os.path.join(chunk_dir, "gold_baseline_Type_2_Diabetes_NoNA.txt")


# =============================================================================
# Step 1: Load chunk files
# =============================================================================
chunk_files = sorted(glob.glob(chunk_glob))
if not chunk_files:
    raise FileNotFoundError(f"No files found matching: {chunk_glob}")

print(f"Found {len(chunk_files)} chunk files")


# =============================================================================
# Step 2: Process each chunk — NO per-chunk dedup (matches AURUM)
# =============================================================================
def process_chunk(file, chunk_idx=0):
    df = pd.read_csv(file, sep="\t", dtype=str)

    print(f"DBG| [CHUNK_{chunk_idx}] file={os.path.basename(file)}  rows={len(df):,}")
    if chunk_idx == 0:
        print(f"DBG| [CHUNK_0] columns={list(df.columns)}")

    for req_col in ["patid", "eventdate", "medcode"]:
        if req_col not in df.columns:
            print(f"DBG| [CHUNK_{chunk_idx}] ⚠ MISSING COLUMN '{req_col}'! Available: {list(df.columns)}")

    df = df[["patid", "eventdate", "medcode"]]

    # ── DATE PARSING DIAGNOSTICS ──
    if chunk_idx < 3:
        print(f"DBG| [CHUNK_{chunk_idx}_DATE_RAW] eventdate dtype={df['eventdate'].dtype}")
        print(f"DBG| [CHUNK_{chunk_idx}_DATE_RAW] eventdate sample(5): {df['eventdate'].dropna().head(5).tolist()}")
        problematic = df['eventdate'].dropna().astype(str)
        has_9999 = problematic.str.contains('9999', na=False).sum()
        has_1800s = problematic.str.contains(r'18[0-6]\d', na=False).sum()
        print(f"DBG| [CHUNK_{chunk_idx}_DATE_RAW] contains '9999': {has_9999:,}  "
              f"contains 18xx: {has_1800s:,}")

    before_parse_missing = df['eventdate'].isna().sum()
    before_parse_patids = set(df['patid'].unique())

    # ── PARSE DATES ──
    df["eventdate"] = pd.to_datetime(df["eventdate"], errors='coerce', dayfirst=True)

    nat_after = df["eventdate"].isna().sum()
    new_nats = nat_after - before_parse_missing
    print(f"DBG| [CHUNK_{chunk_idx}_PARSE] eventdate: was_missing={before_parse_missing:,} "
          f"NaT_after_coerce={nat_after:,} new_NaT_from_parse={new_nats:,}")

    if nat_after > 0 and chunk_idx < 3:
        valid_yrs = df["eventdate"].dropna().dt.year
        if len(valid_yrs) > 0:
            print(f"DBG| [CHUNK_{chunk_idx}_PARSE] parsed year: min={valid_yrs.min():.0f} "
                  f"p50={valid_yrs.median():.0f} max={valid_yrs.max():.0f} "
                  f">2025={(valid_yrs>2025).sum():,} <1900={(valid_yrs<1900).sum():,}")

    # ── SUPERVISOR REQUEST: patients with missing dates BEFORE dropna ──
    patids_missing_before = set(df.loc[df['eventdate'].isna(), 'patid'].unique())

    # ── DROP NA DATES (matches AURUM) ──
    df = df.dropna(subset=["eventdate"])

    after_dropna_patids = set(df['patid'].unique())
    print(f"DBG| [CHUNK_{chunk_idx}_DROPNA] rows: {len(df) + (nat_after):,} → {len(df):,} "
          f"(dropped {nat_after:,})")
    dbg_set_diff(before_parse_patids, after_dropna_patids, f"CHUNK_{chunk_idx}_DROPNA_PATIDS")

    fully_lost = patids_missing_before - after_dropna_patids
    if fully_lost:
        print(f"DBG| [CHUNK_{chunk_idx}_DATE_LOSS] ⚠ {len(fully_lost):,} patients "
              f"FULLY lost because ALL their rows had bad/missing dates")
        print(f"DBG|   sample (≤20): {sorted(fully_lost)[:20]}")

    return df


all_chunks = []
for i, f in enumerate(chunk_files):
    all_chunks.append(process_chunk(f, chunk_idx=i))

all_data = pd.concat(all_chunks, ignore_index=True)
del all_chunks

dbg(all_data, "ALL_CHUNKS_COMBINED", id_col="patid", date_cols=["eventdate"])


# =============================================================================
# Step 3: Map medcodes to diabetes types (matches AURUM)
# =============================================================================
codes_df = pd.read_csv(codes_file, sep="\t", dtype=str)

code_col = None
for c in ["code", "medcode", "medcodeid"]:
    if c in codes_df.columns:
        code_col = c
        break
if code_col is None:
    raise KeyError(f"No medcode column found. Available columns: {codes_df.columns.tolist()}")

print(f"DBG| [CODES_MAP] using column '{code_col}', unique codes={codes_df[code_col].nunique():,}")

codes_df["type"] = codes_df["type"].replace({"0": "2"})

medcode_to_type = codes_df.set_index(code_col)["type"].to_dict()

before_map_patids = set(all_data['patid'].unique())
before_map_rows = len(all_data)

all_data["diabetes_type"] = all_data["medcode"].map(medcode_to_type)

unmapped = all_data["diabetes_type"].isna().sum()
print(f"DBG| [MAP_TYPE] unmapped (medcode not in codelist): {unmapped:,}/{len(all_data):,} "
      f"({100*unmapped/max(len(all_data),1):.1f}%)")
if unmapped > 0:
    unmapped_codes = all_data.loc[all_data["diabetes_type"].isna(), "medcode"].unique()
    print(f"DBG| [MAP_TYPE] unmapped medcodes sample (≤20): {sorted(unmapped_codes)[:20]}")
    print(f"DBG| [MAP_TYPE] data medcode sample: {all_data['medcode'].dropna().head(5).tolist()}")
    print(f"DBG| [MAP_TYPE] codelist key sample:   {list(medcode_to_type.keys())[:5]}")

# ── Filter to type 1 & 2 (matches AURUM — BEFORE dedup) ──
all_data = all_data[all_data["diabetes_type"].isin(["1", "2"])]

after_type_patids = set(all_data['patid'].unique())
print(f"DBG| [TYPE_FILTER] rows: {before_map_rows:,} → {len(all_data):,}")
dbg(all_data, "AFTER_TYPE_FILTER", id_col="patid")
dbg_set_diff(before_map_patids, after_type_patids, "TYPE_FILTER_PATIDS")


# =============================================================================
# Step 4: Save ungrouped baseline DataFrame
# =============================================================================
all_data.to_csv(out_baseline_ungrouped, sep="\t", index=False)
print(f"Ungrouped DataFrame saved as '{out_baseline_ungrouped}'")
dbg(all_data, "UNGROUPED_SAVED", id_col="patid", date_cols=["eventdate"])


# =============================================================================
# Step 5: Derive indexdate per patient (earliest diagnosis — matches AURUM)
# =============================================================================
pre_group_missing_date_patids = set(all_data.loc[all_data['eventdate'].isna(), 'patid'].unique())
pre_group_patids = set(all_data['patid'].unique())
print(f"DBG| [PRE_GROUPBY] patids={len(pre_group_patids):,}  "
      f"patids_with_missing_eventdate={len(pre_group_missing_date_patids):,}")

recs_per_pat = all_data.groupby('patid').size()
print(f"DBG| [PRE_GROUPBY] records_per_patient: min={recs_per_pat.min()} "
      f"p50={recs_per_pat.median():.0f} max={recs_per_pat.max()} "
      f"single_record={(recs_per_pat==1).sum():,}")

type_per_patient = all_data.groupby('patid')['diabetes_type'].nunique()
mixed_type_patients = (type_per_patient > 1).sum()
print(f"DBG| [PRE_GROUPBY] patients with MIXED type codes (both 1 & 2): {mixed_type_patients:,}")

grouped_df = (
    all_data.sort_values(by=["patid", "eventdate"])
    .drop_duplicates(subset="patid", keep="first")
    .copy()
)
grouped_df["indexdate"] = grouped_df["eventdate"]

post_group_missing_date_patids = set(grouped_df.loc[grouped_df['eventdate'].isna(), 'patid'].unique())
post_group_patids = set(grouped_df['patid'].unique())

print(f"DBG| [POST_GROUPBY] patids={len(post_group_patids):,}  "
      f"patids_with_missing_eventdate={len(post_group_missing_date_patids):,}")

missing_only_before = pre_group_missing_date_patids - post_group_missing_date_patids
missing_only_after  = post_group_missing_date_patids - pre_group_missing_date_patids
missing_both        = pre_group_missing_date_patids & post_group_missing_date_patids

print(f"DBG| [GROUPBY_DATE_COMPARE] missing_date patients:")
print(f"DBG|   before_only={len(missing_only_before):,}  "
      f"after_only={len(missing_only_after):,}  "
      f"both={len(missing_both):,}")

dbg_set_diff(pre_group_patids, post_group_patids, "GROUPBY_PATIDS")
dbg(grouped_df, "GROUPED", id_col="patid", date_cols=["eventdate", "indexdate"])

print(f"DBG| [POST_GROUPBY] type distribution: {grouped_df['diabetes_type'].value_counts().to_dict()}")

grouped_df.to_csv(out_baseline_grouped, sep="\t", index=False)
print(f"Grouped DataFrame saved as '{out_baseline_grouped}'")


# =============================================================================
# Step 6: Save separate files for Type 1 and Type 2
# =============================================================================
type_1_df = grouped_df[grouped_df["diabetes_type"] == "1"]
type_2_df = grouped_df[grouped_df["diabetes_type"] == "2"]

print(f"\nSummary of Diabetes Types:")
print(f"Total Type 1 Diabetes patients: {type_1_df['patid'].nunique()}")
print(f"Total Type 2 Diabetes patients: {type_2_df['patid'].nunique()}")

dbg(type_1_df, "TYPE1_FINAL", id_col="patid", date_cols=["indexdate"])
dbg(type_2_df, "TYPE2_FINAL", id_col="patid", date_cols=["indexdate"])

type_1_df.to_csv(out_type1, sep="\t", index=False)
type_2_df.to_csv(out_type2, sep="\t", index=False)

print(f"\nSaved:")
print(f"  - {out_type1}")
print(f"  - {out_type2}")
print("\nDONE.")
