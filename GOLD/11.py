#!/usr/bin/env python3
"""
GOLD Medication Extraction & Cleaning
======================================
Reads harmonised medication codelist, joins to filtered GOLD Therapy files,
and produces patient-level binary indicators:

  prescribed = 1 if any prescription on or before index date, else 0

Outputs:
  - GOLD_medication_long.csv        (patid × med_class, dates, counts)
  - GOLD_medication_wide.csv        (one row per patient, binary columns)
  - GOLD_medication_qc_summary.csv  (prevalence checks per class)
"""

import os
import glob
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Unified medication codelist (code = GOLD prodcode)
CODELIST_PATH = "/scratch/alice/b/bg205/Medication_Extraction/final_outputs/master_codelists/Final_codelists/GOLD_all_medication_lookup_stage3.csv"

# Combined cohort file (will filter to GOLD only)
COHORT_PATH = "/scratch/alice/b/bg205/16_02_26/Combined_GOLD_Aurum_recoded.txt"
COHORT_SEP = "\t"

# Filtered GOLD Therapy files (output from previous step)
THERAPY_DIR = "/scratch/alice/b/bg205/28_02_GOLD/therapy"

# Output directory
OUTPUT_DIR = "/scratch/alice/b/bg205/28_02_GOLD/medication_output"


# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD CODELIST
# ═══════════════════════════════════════════════════════════════════════

def load_codelist(path):
    log.info(f"Loading codelist: {path}")
    cl = pd.read_csv(path, dtype=str)
    cl.columns = cl.columns.str.strip().str.lower()

    for col in ["code", "med_class"]:
        if col not in cl.columns:
            raise KeyError(f"Codelist missing required column '{col}'. Found: {list(cl.columns)}")

    cl["code"] = cl["code"].str.strip()
    cl = cl.dropna(subset=["code", "med_class"])
    cl = cl.drop_duplicates(subset=["code", "med_class"])

    log.info(f"  {len(cl):,} code-class mappings across {cl['med_class'].nunique()} classes")
    for cls, count in cl["med_class"].value_counts().items():
        log.info(f"    {cls}: {count:,} codes")

    # Build lookup: code → set of med_classes
    code_to_classes = cl.groupby("code")["med_class"].apply(set).to_dict()

    return cl, code_to_classes


# ═══════════════════════════════════════════════════════════════════════
# 2. LOAD COHORT
# ═══════════════════════════════════════════════════════════════════════

def load_cohort(path, sep):
    log.info(f"Loading cohort: {path}")
    cohort = pd.read_csv(path, sep=sep, dtype=str)
    cohort.columns = cohort.columns.str.strip().str.lower()

    # Filter to GOLD only
    if "database" in cohort.columns:
        cohort = cohort[cohort["database"].str.upper() == "GOLD"].copy()
        log.info(f"  Filtered to GOLD: {len(cohort):,} patients")
    else:
        log.warning("  No 'database' column — using all patients")

    # Identify indexdate column
    date_col = None
    for candidate in ["indexdate", "index_date"]:
        if candidate in cohort.columns:
            date_col = candidate
            break
    if date_col is None:
        raise KeyError(f"No indexdate column found. Available: {list(cohort.columns)}")

    cohort["indexdate"] = pd.to_datetime(cohort[date_col], errors="coerce")
    n_missing = cohort["indexdate"].isna().sum()
    if n_missing > 0:
        log.warning(f"  {n_missing:,} patients with unparseable indexdate — dropping them")
        cohort = cohort.dropna(subset=["indexdate"])

    cohort = cohort[["patid", "indexdate"]].drop_duplicates(subset=["patid"])
    log.info(f"  Final cohort: {len(cohort):,} unique GOLD patients")

    return cohort


# ═══════════════════════════════════════════════════════════════════════
# 3. EXTRACT PRESCRIPTIONS
# ═══════════════════════════════════════════════════════════════════════

def extract_prescriptions(therapy_dir, codelist_df, cohort):
    """
    Read each filtered Therapy file, inner-join on prodcode to codelist,
    merge indexdate from cohort, filter to <= indexdate.
    Returns long DataFrame of matched prescriptions.
    """
    pattern = os.path.join(therapy_dir, "Cleaned_GOLD_Therapy_*.txt")
    files = sorted(glob.glob(pattern))
    assert len(files) > 0, f"No therapy files found in {therapy_dir}"
    log.info(f"Found {len(files)} filtered Therapy files")

    # Prepare codelist join key
    lookup = codelist_df[["code", "med_class"]].copy()
    lookup["code"] = lookup["code"].str.strip()

    # Cohort as dict for fast lookup
    cohort_dict = cohort.set_index("patid")["indexdate"].to_dict()
    cohort_patids = set(cohort["patid"])

    all_matched = []
    total_rows = 0
    total_matched = 0

    for i, fpath in enumerate(files, start=1):
        df = pd.read_csv(fpath, sep="\t", dtype=str)
        df.columns = df.columns.str.strip().str.lower()
        total_rows += len(df)

        if "prodcode" not in df.columns:
            log.warning(f"  File {i}: no 'prodcode' column — skipping")
            continue

        # Keep only cohort patients
        df = df[df["patid"].isin(cohort_patids)].copy()

        # Inner join to codelist on prodcode
        df["prodcode"] = df["prodcode"].str.strip()
        matched = df.merge(lookup, left_on="prodcode", right_on="code", how="inner")

        if len(matched) == 0:
            log.info(f"  File {i}/{len(files)}: {os.path.basename(fpath)} — 0 matches")
            continue

        # Parse eventdate and get indexdate
        matched["eventdate"] = pd.to_datetime(matched["eventdate"], dayfirst=True, errors="coerce")
        matched["indexdate"] = matched["patid"].map(cohort_dict)

        # Filter: prescription on or before index date
        before = len(matched)
        matched = matched[matched["eventdate"] <= matched["indexdate"]]

        total_matched += len(matched)
        log.info(f"  File {i}/{len(files)}: {os.path.basename(fpath)} — "
                 f"{before:,} code matches, {len(matched):,} on/before indexdate")

        all_matched.append(matched[["patid", "med_class", "eventdate"]])

    if not all_matched:
        log.warning("No matched prescriptions found across all files!")
        return pd.DataFrame(columns=["patid", "med_class", "eventdate"])

    result = pd.concat(all_matched, ignore_index=True)
    log.info(f"\nTotal therapy rows scanned: {total_rows:,}")
    log.info(f"Total matched prescriptions on/before indexdate: {len(result):,}")
    log.info(f"Unique patients with ≥1 match: {result['patid'].nunique():,}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# 4. COLLAPSE TO LONG FORMAT
# ═══════════════════════════════════════════════════════════════════════

def collapse_to_long(matched_df):
    """
    Collapse matched prescriptions to one row per patid × med_class.
    """
    if matched_df.empty:
        return pd.DataFrame(columns=["patid", "med_class", "prescribed",
                                     "first_rx_date", "last_rx_date", "n_prescriptions"])

    long = (
        matched_df
        .groupby(["patid", "med_class"])
        .agg(
            first_rx_date=("eventdate", "min"),
            last_rx_date=("eventdate", "max"),
            n_prescriptions=("eventdate", "count"),
        )
        .reset_index()
    )
    long["prescribed"] = 1

    log.info(f"Long format: {len(long):,} rows (patid × med_class combinations)")
    return long


# ═══════════════════════════════════════════════════════════════════════
# 5. PIVOT TO WIDE FORMAT
# ═══════════════════════════════════════════════════════════════════════

def pivot_to_wide(long_df, cohort, all_classes):
    """
    One row per patient. Columns: {med_class}_prescribed (0/1).
    Patients with no prescriptions get 0 for all classes.
    """
    if long_df.empty:
        wide = cohort[["patid"]].copy()
        for cls in all_classes:
            wide[f"{cls}_prescribed"] = 0
        return wide

    # Pivot
    wide = long_df.pivot_table(
        index="patid",
        columns="med_class",
        values="prescribed",
        aggfunc="max",
        fill_value=0,
    ).rename(columns=lambda c: f"{c}_prescribed").reset_index()

    # Left join to full cohort so unprescribed patients get 0s
    wide = cohort[["patid"]].merge(wide, on="patid", how="left")

    # Ensure all classes present even if zero patients matched
    for cls in all_classes:
        col = f"{cls}_prescribed"
        if col not in wide.columns:
            wide[col] = 0

    # Fill NaN with 0
    prescribed_cols = [c for c in wide.columns if c.endswith("_prescribed")]
    wide[prescribed_cols] = wide[prescribed_cols].fillna(0).astype(int)

    # Reorder columns
    ordered = ["patid"] + sorted(prescribed_cols)
    wide = wide[ordered]

    log.info(f"Wide format: {len(wide):,} patients × {len(prescribed_cols)} medication columns")
    return wide


# ═══════════════════════════════════════════════════════════════════════
# 6. QC SUMMARY
# ═══════════════════════════════════════════════════════════════════════

def run_qc(wide_df, long_df, cohort):
    n_cohort = len(cohort)
    prescribed_cols = [c for c in wide_df.columns if c.endswith("_prescribed")]

    rows = []
    for col in sorted(prescribed_cols):
        cls = col.replace("_prescribed", "")
        n_prescribed = wide_df[col].sum()
        prevalence = n_prescribed / n_cohort * 100 if n_cohort > 0 else 0

        # Get date info from long if available
        cls_long = long_df[long_df["med_class"] == cls] if not long_df.empty else pd.DataFrame()
        rows.append({
            "med_class": cls,
            "n_prescribed": int(n_prescribed),
            "n_not_prescribed": int(n_cohort - n_prescribed),
            "prevalence_pct": round(prevalence, 2),
            "median_n_rx": cls_long["n_prescriptions"].median() if not cls_long.empty else 0,
            "earliest_rx": cls_long["first_rx_date"].min() if not cls_long.empty else pd.NaT,
            "latest_rx": cls_long["last_rx_date"].max() if not cls_long.empty else pd.NaT,
        })

    qc = pd.DataFrame(rows)

    log.info(f"\n{'='*60}")
    log.info("QC SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"Total GOLD cohort: {n_cohort:,}")
    log.info(f"Patients with ≥1 prescription (any class): "
             f"{(wide_df[prescribed_cols].sum(axis=1) > 0).sum():,}")
    log.info(f"\n{qc.to_string(index=False)}")

    return qc


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load codelist
    codelist_df, code_to_classes = load_codelist(CODELIST_PATH)
    all_classes = sorted(codelist_df["med_class"].unique())

    # 2. Load cohort
    cohort = load_cohort(COHORT_PATH, COHORT_SEP)

    # 3. Extract prescriptions
    matched = extract_prescriptions(THERAPY_DIR, codelist_df, cohort)

    # 4. Collapse to long
    long_df = collapse_to_long(matched)

    # 5. Pivot to wide
    wide_df = pivot_to_wide(long_df, cohort, all_classes)

    # 6. QC
    qc_df = run_qc(wide_df, long_df, cohort)

    # 7. Save
    long_path = os.path.join(OUTPUT_DIR, "GOLD_medication_long.csv")
    wide_path = os.path.join(OUTPUT_DIR, "GOLD_medication_wide.csv")
    qc_path   = os.path.join(OUTPUT_DIR, "GOLD_medication_qc_summary.csv")

    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    qc_df.to_csv(qc_path, index=False)

    log.info(f"\nSaved:")
    log.info(f"  Long:  {long_path}")
    log.info(f"  Wide:  {wide_path}")
    log.info(f"  QC:    {qc_path}")
    log.info("Done.")


if __name__ == "__main__":
    main()
