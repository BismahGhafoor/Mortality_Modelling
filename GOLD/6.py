#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SCRIPT 6 (SUPERVISOR-ALIGNED + WINDOW RULES) — FIXED INDEXDATE PARSING
"""

import os
import glob
import re
import numpy as np
import pandas as pd

print("starting...")

from helper_functions import lcf, ucf, perc
from helper_functions import save_long_format_data, read_long_format_data
from helper_functions import remap_eth, nperc_counts, calc_gfr

save_long_format = False


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


def dbg_patid_diff(before_patids, after_patids, step_name):
    lost = before_patids - after_patids
    gained = after_patids - before_patids
    print(f"DBG|[{step_name}] patids before={len(before_patids):,}  after={len(after_patids):,}  "
          f"lost={len(lost):,}  gained={len(gained):,}")
    if lost and len(lost) <= 20:
        print(f"DBG|  lost (first 20): {sorted(lost)[:20]}")
    elif lost:
        print(f"DBG|  lost (first 20): {sorted(lost)[:20]}")


def dbg_filter(df_before, df_after, step_name, date_cols=None):
    """Quick before/after row+patid comparison for a filter step."""
    r_before, r_after = len(df_before), len(df_after)
    p_before = df_before['patid'].nunique() if 'patid' in df_before.columns else 0
    p_after = df_after['patid'].nunique() if 'patid' in df_after.columns and len(df_after) > 0 else 0
    print(f"DBG|[{step_name}] rows: {r_before:,} -> {r_after:,} (Δ{r_after-r_before:+,})  "
          f"patids: {p_before:,} -> {p_after:,} (Δ{p_after-p_before:+,})")
    if date_cols and len(df_after) > 0:
        for c in date_cols:
            if c in df_after.columns:
                print(f"DBG|  {c} missing after: {df_after[c].isna().sum():,}")


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BASELINE_FILE = "/scratch/alice/b/bg205/28_02_GOLD/Enriched_baseline_with_demographics.txt"
CLINICAL_SMOK_FILE = "/scratch/alice/b/bg205/28_02_GOLD/Clinical_SmokingStatus_all.txt.gz"

CLINICAL_ZIP_PATTERN = "/scratch/alice/b/bg205/DataCleaning_Gold_v2/GOLD/FZ_GOLD_All_Extract_Clinical_*.zip"
CLINICAL_TXT_PATTERN = "/scratch/alice/b/bg205/DataCleaning_Gold_v2/GOLD/FZ_GOLD_All_Extract_Clinical_*.txt"

ADD_ZIP_PATTERN = "/scratch/alice/b/bg205/DataCleaning_Gold_v2/GOLD/FZ_GOLD_All_Extract_Additional_*.zip"
ADD_TXT_PATTERN = "/scratch/alice/b/bg205/DataCleaning_Gold_v2/GOLD/FZ_GOLD_All_Extract_Additional_*.txt"

HES_FILE = "/scratch/alice/b/bg205/DataCleaning_Gold_v2/GOLD_linked/hes_diagnosis_hosp_23_002869_DM.txt"

CODES_XLSX = "/scratch/alice/b/bg205/DataCleaning_FINAL_Gold/GOLD_Codes_FZ.xlsx"
CODES_SHEET = "Smok"

OUTPUT_FILE = "/scratch/alice/b/bg205/28_02_GOLD/Cleaned_Patient_Smoking_BMI_BP_Data_3YEAR.txt"

CHUNKSIZE = 20000
WINDOW_DAYS = 1095
#WINDOW_DAYS = 365

MIN_DATE = pd.Timestamp("1900-01-01")
MAX_DATE = pd.Timestamp("2024-12-31")

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def norm_code(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def parse_indexdate_iso(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="%Y-%m-%d")

def ensure_indexdate_only(df, patient):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["patid"] = df["patid"].astype(str)

    if "indexdate" not in df.columns:
        before_rows = len(df)
        df = df.merge(patient[["patid", "indexdate"]], on="patid", how="left")
        print(f"DBG|[ensure_indexdate_only] merged indexdate: rows {before_rows:,} -> {len(df):,}  "
              f"indexdate_missing_after_merge={df['indexdate'].isna().sum():,}")

    df["indexdate"] = parse_indexdate_iso(df["indexdate"])
    return df


def attach_eventdate_from_clinical(add_df, clinical_lookup, patient, label="additional"):
    """
    Recover real eventdate for Additional rows by linking to Clinical on
    patid + adid + enttype, then merge indexdate from patient.
    """
    if add_df is None or add_df.empty:
        return pd.DataFrame()

    x = add_df.copy()
    x["patid"] = x["patid"].astype(str)
    x["adid"] = x["adid"].astype(str)
    x["enttype"] = x["enttype"].astype(str)

    before_rows = len(x)
    before_patids = x["patid"].nunique()

    x = x.merge(
        clinical_lookup[["patid", "adid", "enttype", "eventdate"]],
        on=["patid", "adid", "enttype"],
        how="left"
    )

    x = x.merge(patient[["patid", "indexdate"]], on="patid", how="left")

    x["eventdate"] = pd.to_datetime(x["eventdate"], errors="coerce", dayfirst=True)
    x = enforce_date_bounds(x, "eventdate")
    x["indexdate"] = parse_indexdate_iso(x["indexdate"])

    print(f"DBG|[attach_eventdate_from_clinical:{label}] rows={before_rows:,} patids={before_patids:,}")
    print(f"DBG|[attach_eventdate_from_clinical:{label}] eventdate missing after merge={x['eventdate'].isna().sum():,}")
    print(f"DBG|[attach_eventdate_from_clinical:{label}] indexdate missing after merge={x['indexdate'].isna().sum():,}")

    return x

def pick_closest_before_or_at(df, value_col, date_col="eventdate", group_cols=("patid", "indexdate")):
    if df is None or df.empty:
        return pd.DataFrame(columns=list(group_cols) + [date_col, value_col])

    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x["indexdate"] = parse_indexdate_iso(x["indexdate"])

    # ── DBG: before filters ──
    dbg(x, f"pick_closest_{value_col}_input", date_cols=[date_col, "indexdate"])

    before_filter = set(x['patid'].unique())
    x = x[x[date_col].notna() & x["indexdate"].notna() & x[value_col].notna()]
    after_notna = set(x['patid'].unique())
    print(f"DBG|[pick_closest_{value_col}] after dropna(date+index+value): "
          f"rows={len(x):,}  patids lost={len(before_filter - after_notna):,}")

    before_window = set(x['patid'].unique())
    x = x[x[date_col] <= x["indexdate"]]
    x["gap_days"] = (x["indexdate"] - x[date_col]).dt.days
    x = x[(x["gap_days"] >= 0) & (x["gap_days"] <= WINDOW_DAYS)]
    
    after_window = set(x['patid'].unique())
    print(f"DBG|[pick_closest_{value_col}] after eventdate<=indexdate: "
          f"rows={len(x):,}  patids lost={len(before_window - after_window):,}")

    if x.empty:
        return pd.DataFrame(columns=list(group_cols) + [date_col, value_col])

    x["gap_days"] = (x["indexdate"] - x[date_col]).dt.days
    x = x.sort_values(list(group_cols) + ["gap_days", date_col], ascending=[True, True, True, False])
    best = x.drop_duplicates(subset=list(group_cols), keep="first").reset_index(drop=True)

    print(f"DBG|[pick_closest_{value_col}] final best: rows={len(best):,}  patids={best['patid'].nunique():,}  "
          f"gap_days: min={best['gap_days'].min()}  p50={best['gap_days'].median():.0f}  max={best['gap_days'].max()}")

    return best[list(group_cols) + [date_col, value_col]]

def pick_closest_within_window_before_or_at(df, value_cols, window_days=1095, date_col="eventdate", group_cols=("patid", "indexdate")):
    if df is None or df.empty:
        return pd.DataFrame(columns=list(group_cols) + [date_col] + list(value_cols))

    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x["indexdate"] = parse_indexdate_iso(x["indexdate"])

    # ── DBG ──
    label = "_".join(value_cols)
    dbg(x, f"pick_window_{label}_input", date_cols=[date_col, "indexdate"])

    before_filter = set(x['patid'].unique())
    x = x[x[date_col].notna() & x["indexdate"].notna()]
    x = x[x[date_col] <= x["indexdate"]]
    after_leq = set(x['patid'].unique())
    print(f"DBG|[pick_window_{label}] after date<=indexdate: "
          f"rows={len(x):,}  patids lost={len(before_filter - after_leq):,}")

    if x.empty:
        return pd.DataFrame(columns=list(group_cols) + [date_col] + list(value_cols))

    x["gap_days"] = (x["indexdate"] - x[date_col]).dt.days
    before_window = set(x['patid'].unique())
    x = x[(x["gap_days"] >= 0) & (x["gap_days"] <= window_days)]
    after_window = set(x['patid'].unique())
    print(f"DBG|[pick_window_{label}] after {window_days}d window: "
          f"rows={len(x):,}  patids lost by window={len(before_window - after_window):,}")

    if x.empty:
        return pd.DataFrame(columns=list(group_cols) + [date_col] + list(value_cols))

    x = x.sort_values(list(group_cols) + ["gap_days", date_col], ascending=[True, True, True, False])
    best = x.drop_duplicates(subset=list(group_cols), keep="first").reset_index(drop=True)

    print(f"DBG|[pick_window_{label}] final best: rows={len(best):,}  patids={best['patid'].nunique():,}  "
          f"gap_days: min={best['gap_days'].min()}  p50={best['gap_days'].median():.0f}  max={best['gap_days'].max()}")

    keep_cols = ["patid", "indexdate", date_col] + list(value_cols)
    return best[keep_cols]

def enforce_date_bounds(df, date_col):
    dt = df[date_col]
    before = len(df)
    df = df[(dt.notna()) & (dt >= MIN_DATE) & (dt <= MAX_DATE)]
    after = len(df)
    if before - after > 0:
        print(f"DBG|[DATE_BOUNDS] {date_col}: dropped {before - after:,} rows outside "
              f"[{MIN_DATE.date()}, {MAX_DATE.date()}]  kept={after:,}")
    return df
# ----------------------------------------------------------------------
# Smoking processing
# ----------------------------------------------------------------------
def get_smoking_data(smoking_data, clinical_smok, patient, hes_hosp):
    smoking_data["patid"] = smoking_data["patid"].astype(str)
    clinical_smok["patid"] = clinical_smok["patid"].astype(str)
    patient["patid"] = patient["patid"].astype(str)

    print(f"\n{'='*60}")
    print("SMOKING PROCESSING")
    print(f"{'='*60}")

    # --- Additional (enttype 4) ---
    print("\n--- Smoking: Additional (enttype 4) ---")
    print("Columns in smoking_data before subsetting:", smoking_data.columns.tolist())
    smok_add = smoking_data[["patid", "indexdate", "eventdate", "data1"]].copy()
    smok_add["data1"] = pd.to_numeric(smok_add["data1"], errors="coerce")

    dbg(smok_add, "smok_add_raw", date_cols=["eventdate", "indexdate"])
    print(f"DBG|[smok_add] data1 value_counts:\n{smok_add['data1'].value_counts(dropna=False).head(10)}")

    before_pats = set(smok_add['patid'].unique())
    smok_add = smok_add[(smok_add["data1"].isin([1, 2, 3])) & smok_add["eventdate"].notnull()]
    dbg_filter_pats = set(smok_add['patid'].unique())
    print(f"DBG|[smok_add_filter_data1] patids: {len(before_pats):,} -> {len(dbg_filter_pats):,}  "
          f"lost={len(before_pats - dbg_filter_pats):,}")

    smok_add["eventdate"] = pd.to_datetime(smok_add["eventdate"], errors="coerce", dayfirst=True)
    smok_add = enforce_date_bounds(smok_add, "eventdate")
    smok_add["indexdate"] = parse_indexdate_iso(smok_add["indexdate"])

    before_dropna = set(smok_add['patid'].unique())
    smok_add = smok_add.dropna(subset=["eventdate", "indexdate"])
    after_dropna = set(smok_add['patid'].unique())
    print(f"DBG|[smok_add_dropna_dates] patids: {len(before_dropna):,} -> {len(after_dropna):,}  "
          f"lost={len(before_dropna - after_dropna):,}")

    smok_add = smok_add.sort_values(["patid", "eventdate"]).reset_index(drop=True)
    smok_add = smok_add.sample(frac=1).drop_duplicates(subset=["patid", "eventdate"], keep="last")
    smok_add["data1"] = smok_add["data1"].replace({1: "Yes", 2: "No", 3: "Ex"})
    nperc_counts(smok_add, "data1")
    smok_add = smok_add.rename(columns={"data1": "smok_add"})

    dbg(smok_add, "smok_add_final", date_cols=["eventdate"])

    # --- Clinical (medcodes) ---
    print("\n--- Smoking: Clinical (medcodes) ---")
    smok_clref = clinical_smok.copy(deep=True)
    if "indexdate" not in smok_clref.columns:
        before_merge = len(smok_clref)
        smok_clref = smok_clref.merge(patient[["patid", "indexdate"]], on="patid", how="left")
        n_no_index = smok_clref['indexdate'].isna().sum()
        print(f"DBG|[smok_clref_merge_index] rows: {before_merge:,} -> {len(smok_clref):,}  "
              f"indexdate_missing={n_no_index:,}")

    smok_clref["eventdate"] = pd.to_datetime(smok_clref["eventdate"], errors="coerce", dayfirst=True)
    smok_clref = enforce_date_bounds(smok_clref, "eventdate")
    smok_clref["indexdate"] = parse_indexdate_iso(smok_clref["indexdate"])
    smok_clref["medcode"] = smok_clref["medcode"].map(norm_code)

    smok_clref["smok_clref"] = np.nan
    smoke_keys = ["current smoker", "never smoker", "ex smoker"]
    smoke_cat = ["Yes", "No", "Ex"]

    smoke_codes = pd.read_excel(CODES_XLSX, sheet_name=CODES_SHEET, dtype=str)
    smoke_codes["medcode"] = smoke_codes["medcode"].map(norm_code)
    smoke_codes = smoke_codes[smoke_codes["source"] == "cprd"]

    smoke_dict = {k: smoke_codes.loc[smoke_codes["type"] == k, "medcode"].dropna().tolist() for k in smoke_keys}
    print(f"DBG|[smoke_codes] categories: " +
          ", ".join(f"{k}={len(v)} codes" for k, v in smoke_dict.items()))

    for i, key in enumerate(smoke_keys):
        smok_clref.loc[smok_clref["medcode"].isin(smoke_dict.get(key, [])), "smok_clref"] = smoke_cat[i]

    # ── DBG: how many clinical records mapped to a smoking category? ──
    n_mapped = smok_clref["smok_clref"].notna().sum()
    n_unmapped = smok_clref["smok_clref"].isna().sum()
    print(f"DBG|[smok_clref_mapping] mapped={n_mapped:,}  unmapped(dropped)={n_unmapped:,}")
    if n_unmapped > 0:
        unmapped_codes = smok_clref.loc[smok_clref["smok_clref"].isna(), "medcode"].value_counts().head(10)
        print(f"DBG|  top unmapped medcodes:\n{unmapped_codes}")

    before_pats_cl = set(smok_clref['patid'].unique())
    smok_clref = smok_clref[smok_clref["smok_clref"].notnull()]
    after_pats_cl = set(smok_clref['patid'].unique())
    print(f"DBG|[smok_clref_filter_mapped] patids: {len(before_pats_cl):,} -> {len(after_pats_cl):,}  "
          f"lost={len(before_pats_cl - after_pats_cl):,}")

    smok_clref = smok_clref.sort_values(["patid", "eventdate"]).reset_index(drop=True)
    smok_clref = smok_clref.sample(frac=1).drop_duplicates(subset=["patid", "eventdate"], keep="last")
    smok_clref = smok_clref[["patid", "indexdate", "eventdate", "smok_clref"]]

    dbg(smok_clref, "smok_clref_final", date_cols=["eventdate"])

    # --- HES (ICD) ---
    print("\n--- Smoking: HES (ICD) ---")
    hes_codes = pd.read_excel(CODES_XLSX, sheet_name=CODES_SHEET, dtype=str)
    hes_codes = hes_codes[hes_codes["source"] == "hes"]["medcode"].dropna().astype(str).str.strip().tolist()
    print(f"DBG|[smok_hes_codes] {len(hes_codes)} ICD codes loaded: {hes_codes[:10]}")

    if hes_codes:
        pat = r"^(?:" + r"|".join(re.escape(c) for c in hes_codes) + r")"
        mask = hes_hosp["ICD"].fillna("").str.contains(pat, regex=True, na=False)
        smok_hes = hes_hosp[mask].copy()
    else:
        smok_hes = hes_hosp.iloc[0:0].copy()

    dbg(smok_hes, "smok_hes_icd_matched", date_cols=["admidate"] if "admidate" in smok_hes.columns else None)

    smok_hes = smok_hes.sort_values(["patid", "admidate"]).reset_index(drop=True)
    smok_hes = smok_hes.sample(frac=1).drop_duplicates(subset=["patid", "admidate"], keep="last")
    smok_hes["smok_hes"] = "Yes"
    smok_hes = smok_hes.rename(columns={"admidate": "eventdate"})

    if "indexdate" not in smok_hes.columns:
        smok_hes = smok_hes.merge(patient[["patid", "indexdate"]], on="patid", how="left")

    smok_hes["eventdate"] = pd.to_datetime(smok_hes["eventdate"], errors="coerce", dayfirst=True)
    smok_hes = enforce_date_bounds(smok_hes, "eventdate")
    smok_hes["indexdate"] = parse_indexdate_iso(smok_hes["indexdate"])

    before_hes_dropna = set(smok_hes['patid'].unique())
    smok_hes = smok_hes.dropna(subset=["eventdate", "indexdate"])
    after_hes_dropna = set(smok_hes['patid'].unique())
    print(f"DBG|[smok_hes_dropna] patids: {len(before_hes_dropna):,} -> {len(after_hes_dropna):,}  "
          f"lost={len(before_hes_dropna - after_hes_dropna):,}")

    smok_hes = smok_hes[["patid", "indexdate", "eventdate", "smok_hes"]]
    dbg(smok_hes, "smok_hes_final", date_cols=["eventdate"])

    # --- Merge preference: clinical → HES → additional ---
    print("\n--- Smoking: Merge all sources ---")
    smoking = smok_clref.merge(smok_add, how="outer", on=["patid", "indexdate", "eventdate"])
    smoking = smoking.merge(smok_hes, how="outer", on=["patid", "indexdate", "eventdate"])

    dbg(smoking, "smoking_merged_all_sources", date_cols=["eventdate", "indexdate"])
    print(f"DBG|[smoking_merged] source coverage:")
    print(f"DBG|  smok_clref non-null: {smoking['smok_clref'].notna().sum():,}")
    print(f"DBG|  smok_hes non-null:   {smoking['smok_hes'].notna().sum():,}")
    print(f"DBG|  smok_add non-null:   {smoking['smok_add'].notna().sum():,}")

    smoking["smoking_status"] = smoking["smok_clref"]
    smoking.loc[smoking["smoking_status"].isna() & smoking["smok_hes"].notna(), "smoking_status"] = smoking["smok_hes"]
    smoking.loc[smoking["smoking_status"].isna() & smoking["smok_add"].notna(), "smoking_status"] = smoking["smok_add"]

    print(f"DBG|[smoking_status_combined] value_counts:\n"
          f"{smoking['smoking_status'].value_counts(dropna=False).head(10)}")

    # HARD pick: closest BEFORE/AT indexdate
    print("\n--- Smoking: pick closest before/at indexdate ---")
    smoking_best = pick_closest_within_window_before_or_at(
        smoking[["patid", "indexdate", "eventdate", "smoking_status"]],
        value_cols=["smoking_status"],
        window_days=WINDOW_DAYS,
        date_col="eventdate"
    )

    smoking_best = smoking_best.rename(columns={"eventdate": "smoking_date"})

    # ── DBG: merge back to patient ──
    pats_with_smoking = set(smoking_best['patid'].unique())
    baseline_pats = set(patient['patid'].unique())
    print(f"DBG|[smoking_coverage] baseline={len(baseline_pats):,}  "
          f"have_smoking_before_idx={len(pats_with_smoking & baseline_pats):,}  "
          f"will_be_missing={len(baseline_pats - pats_with_smoking):,}")

    patient = patient.merge(
        smoking_best[["patid", "indexdate", "smoking_date", "smoking_status"]],
        on=["patid", "indexdate"],
        how="left"
    )

    # QC
    tmp = patient[["indexdate", "smoking_date"]].copy()
    tmp["indexdate"] = parse_indexdate_iso(tmp["indexdate"])
    tmp["smoking_date"] = pd.to_datetime(tmp["smoking_date"], errors="coerce", dayfirst=True)
    n_after = int((tmp["smoking_date"] > tmp["indexdate"]).sum())
    n_have = int(tmp["smoking_date"].notna().sum())
    print(f"QC smoking: dates AFTER indexdate = {n_after} / {n_have} (should be 0)")

    print(f"DBG|[smoking_final_on_patient] smoking_status missing="
          f"{patient['smoking_status'].isna().sum():,} / {len(patient):,} "
          f"({patient['smoking_status'].isna().mean()*100:.2f}%)")
    print(f"DBG|[smoking_final_on_patient] value_counts:\n"
          f"{patient['smoking_status'].value_counts(dropna=False)}")

    return patient

# ----------------------------------------------------------------------
# BMI processing
# ----------------------------------------------------------------------
def weight_height_bmi(wh_data, patient):
    print(f"\n{'='*60}")
    print("BMI PROCESSING")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Helper: most recent record on/before indexdate, NO lookback restriction
    # ------------------------------------------------------------------
    def pick_most_recent_before_or_at(df, value_col, date_col="eventdate", group_cols=("patid", "indexdate")):
        if df is None or df.empty:
            return pd.DataFrame(columns=list(group_cols) + [date_col, value_col])

        x = df.copy()
        x[date_col] = pd.to_datetime(x[date_col], errors="coerce", dayfirst=True)
        x["indexdate"] = parse_indexdate_iso(x["indexdate"])

        x = x[x[date_col].notna() & x["indexdate"].notna() & x[value_col].notna()]
        x = x[x[date_col] <= x["indexdate"]]
        if x.empty:
            return pd.DataFrame(columns=list(group_cols) + [date_col, value_col])

        x["gap_days"] = (x["indexdate"] - x[date_col]).dt.days
        x = x.sort_values(list(group_cols) + ["gap_days", date_col], ascending=[True, True, True, False])
        best = x.drop_duplicates(subset=list(group_cols), keep="first").reset_index(drop=True)

        print(f"DBG|[pick_most_recent_{value_col}] rows={len(best):,}  patids={best['patid'].nunique():,}  "
              f"gap_days: min={best['gap_days'].min()}  p50={best['gap_days'].median():.0f}  max={best['gap_days'].max()}")

        return best[list(group_cols) + [date_col, value_col]]

    # ------------------------------------------------------------------
    # Split weight / height / recorded BMI
    # enttype 13 rows contain weight and possibly recorded BMI in data3
    # enttype 14 rows contain height
    # ------------------------------------------------------------------
    weight = wh_data[wh_data["enttype"] == "13"][["patid", "indexdate", "eventdate", "data1"]].rename(
        columns={"data1": "weight_kg"}
    ).copy()

    bmi_recorded = wh_data[wh_data["enttype"] == "13"][["patid", "indexdate", "eventdate", "data3"]].rename(
        columns={"data3": "bmi"}
    ).copy()

    height = wh_data[wh_data["enttype"] == "14"][["patid", "indexdate", "eventdate", "data1"]].rename(
        columns={"data1": "height_m"}
    ).copy()

    print(f"DBG|[bmi_inputs] weight rows={len(weight):,}  recorded_bmi rows={len(bmi_recorded):,}  height rows={len(height):,}")

    # ------------------------------------------------------------------
    # Clean recorded BMI
    # ------------------------------------------------------------------
    bmi_recorded["bmi"] = pd.to_numeric(bmi_recorded["bmi"], errors="coerce")
    bmi_recorded["eventdate"] = pd.to_datetime(bmi_recorded["eventdate"], errors="coerce", dayfirst=True)
    bmi_recorded = enforce_date_bounds(bmi_recorded, "eventdate")
    bmi_recorded["indexdate"] = parse_indexdate_iso(bmi_recorded["indexdate"])

    bmi_recorded = bmi_recorded[(bmi_recorded["bmi"] >= 10) & (bmi_recorded["bmi"] <= 80)]
    bmi_recorded = bmi_recorded.drop_duplicates(subset=["patid", "indexdate", "eventdate", "bmi"], keep="last")
    bmi_recorded["bmi_source"] = "recorded"

    dbg(bmi_recorded, "bmi_recorded_clean", date_cols=["eventdate", "indexdate"])

    print("\n--- BMI: pick closest recorded BMI within lookback window ---")
    bmi_best = pick_closest_within_window_before_or_at(
        bmi_recorded,
        value_cols=["bmi", "bmi_source"],
        window_days=WINDOW_DAYS,
        date_col="eventdate"
    )

    if not bmi_best.empty:
        bmi_best = bmi_best.rename(columns={"eventdate": "bmi_date"})

    recorded_bmi_count = len(bmi_best)
    print(f"DBG|[bmi_recorded_best] recorded_bmi_count={recorded_bmi_count:,}")

    # ------------------------------------------------------------------
    # Clean weight
    # ------------------------------------------------------------------
    weight["weight_kg"] = pd.to_numeric(weight["weight_kg"], errors="coerce")
    weight["eventdate"] = pd.to_datetime(weight["eventdate"], errors="coerce", dayfirst=True)
    weight = enforce_date_bounds(weight, "eventdate")
    weight["indexdate"] = parse_indexdate_iso(weight["indexdate"])

    weight = weight[(weight["weight_kg"] > 0) & (weight["weight_kg"] < 500)]
    weight = weight.drop_duplicates(subset=["patid", "indexdate", "eventdate", "weight_kg"], keep="last")

    print("\n--- BMI: pick closest weight within lookback window ---")
    weight_best = pick_closest_within_window_before_or_at(
        weight,
        value_cols=["weight_kg"],
        window_days=WINDOW_DAYS,
        date_col="eventdate"
    )

    if not weight_best.empty:
        weight_best = weight_best.rename(columns={"eventdate": "weight_date"})

    print(f"DBG|[weight_best] rows={len(weight_best):,}")

    # ------------------------------------------------------------------
    # Clean height
    # ------------------------------------------------------------------
    height["height_m"] = pd.to_numeric(height["height_m"], errors="coerce")
    height["eventdate"] = pd.to_datetime(height["eventdate"], errors="coerce", dayfirst=True)
    height = enforce_date_bounds(height, "eventdate")
    height["indexdate"] = parse_indexdate_iso(height["indexdate"])

    # convert cm to m where needed
    height["height_m"] = np.where(height["height_m"] > 10, height["height_m"] / 100.0, height["height_m"])
    height = height[(height["height_m"] >= 0.5) & (height["height_m"] <= 2.5)]
    height = height.drop_duplicates(subset=["patid", "indexdate", "eventdate", "height_m"], keep="last")

    print("\n--- BMI: pick most recent valid height on/before indexdate (no lookback restriction) ---")
    height_best = pick_most_recent_before_or_at(
        height,
        value_col="height_m",
        date_col="eventdate"
    )

    if not height_best.empty:
        height_best = height_best.rename(columns={"eventdate": "height_date"})

    print(f"DBG|[height_best] rows={len(height_best):,}")

    # ------------------------------------------------------------------
    # Calculated BMI fallback: recent weight + most recent height before indexdate
    # ------------------------------------------------------------------
    calc_best = weight_best.merge(height_best, on=["patid", "indexdate"], how="inner")
    print(f"DBG|[bmi_calc_merge] merged weight_best + height_best rows={len(calc_best):,}")

    if not calc_best.empty:
        calc_best["bmi"] = calc_best["weight_kg"] / (calc_best["height_m"] * calc_best["height_m"])
        calc_best = calc_best[(calc_best["bmi"] >= 10) & (calc_best["bmi"] <= 80)].copy()
        calc_best["bmi_date"] = calc_best["weight_date"]
        calc_best["bmi_source"] = "calculated"

    calc_candidate_count = len(calc_best)
    print(f"DBG|[bmi_calc_candidates] in-range calculated rows={calc_candidate_count:,}")

    # Remove those who already have recorded BMI
    if not bmi_best.empty and not calc_best.empty:
        rec_keys = set(bmi_best["patid"].astype(str) + "_" + bmi_best["indexdate"].astype(str))
        calc_best["key"] = calc_best["patid"].astype(str) + "_" + calc_best["indexdate"].astype(str)
        overlap_count = calc_best["key"].isin(rec_keys).sum()
        calc_best = calc_best[~calc_best["key"].isin(rec_keys)].drop(columns=["key"])
    else:
        overlap_count = 0

    calculated_bmi_count = len(calc_best)
    print(f"DBG|[bmi_calc_fallback] already_have_recorded={overlap_count:,}  new_fallback={calculated_bmi_count:,}")

    # ------------------------------------------------------------------
    # Combine final BMI
    # ------------------------------------------------------------------
    if bmi_best.empty:
        bmi_best = pd.DataFrame(columns=["patid", "indexdate", "bmi_date", "bmi", "bmi_source"])
    else:
        bmi_best = bmi_best[["patid", "indexdate", "bmi_date", "bmi", "bmi_source"]]

    if calc_best.empty:
        calc_best = pd.DataFrame(columns=["patid", "indexdate", "bmi_date", "bmi", "bmi_source"])
    else:
        calc_best = calc_best[["patid", "indexdate", "bmi_date", "bmi", "bmi_source"]]

    bmi_final = pd.concat([bmi_best, calc_best], ignore_index=True).drop_duplicates(["patid", "indexdate"], keep="first")

    print(f"DBG|[bmi_final] recorded={recorded_bmi_count:,}  calculated={calculated_bmi_count:,}  total={len(bmi_final):,}")
    if not bmi_final.empty:
        print(f"DBG|[bmi_final] bmi_source counts:\n{bmi_final['bmi_source'].value_counts(dropna=False)}")

    save_long_format_data(bmi_final, save_long_format, "bmi")

    if not bmi_final.empty:
        patient = patient.merge(
            bmi_final[["patid", "indexdate", "bmi_date", "bmi", "bmi_source"]],
            on=["patid", "indexdate"],
            how="left"
        )
    else:
        patient["bmi_date"] = pd.NaT
        patient["bmi"] = np.nan
        patient["bmi_source"] = np.nan

    print(f"DBG|[bmi_final_on_patient] bmi missing={patient['bmi'].isna().sum():,} / {len(patient):,} "
          f"({patient['bmi'].isna().mean()*100:.2f}%)")
    print(f"DBG|[bmi_final_on_patient] bmi_source value_counts:\n{patient['bmi_source'].value_counts(dropna=False)}")

    return patient

# ----------------------------------------------------------------------
# BP processing
# ----------------------------------------------------------------------
def get_bp_data(bp_data, patient):
    print(f"\n{'='*60}")
    print("BP PROCESSING (SBP ONLY)")
    print(f"{'='*60}")

    """
    # ============================================================
    # OLD PAIRED SBP+DBP LOGIC (commented out)
    # ============================================================
    bp = bp_data[["patid", "eventdate", "indexdate", "data1", "data2"]].rename(
        columns={"data1": "diastolic", "data2": "systolic"}
    ).copy()

    bp["eventdate"] = pd.to_datetime(bp["eventdate"], errors="coerce", dayfirst=True)
    bp = enforce_date_bounds(bp, "eventdate")
    bp["indexdate"] = parse_indexdate_iso(bp["indexdate"])

    bp["diastolic"] = pd.to_numeric(bp["diastolic"], errors="coerce")
    bp["systolic"] = pd.to_numeric(bp["systolic"], errors="coerce")

    dbg(bp, "bp_raw", date_cols=["eventdate", "indexdate"])
    print(f"DBG|[bp_raw] systolic stats: {bp['systolic'].describe().to_dict()}")
    print(f"DBG|[bp_raw] diastolic stats: {bp['diastolic'].describe().to_dict()}")

    before_filter = len(bp)
    pats_before = set(bp.loc[bp['systolic'].notna() | bp['diastolic'].notna(), 'patid'].unique())
    bp = bp[(bp["eventdate"].notnull()) & bp["diastolic"].notnull() & bp["systolic"].notnull()]
    print(f"DBG|[bp_dropna] rows: {before_filter:,} -> {len(bp):,}")

    before_range = len(bp)
    pats_before_range = set(bp['patid'].unique())
    bp = bp[(bp["systolic"] >= 40) & (bp["systolic"] <= 250) & (bp["diastolic"] >= 20) & (bp["diastolic"] <= 200)]
    pats_after_range = set(bp['patid'].unique())
    print(f"DBG|[bp_range_filter] rows: {before_range:,} -> {len(bp):,}  "
          f"patids: {len(pats_before_range):,} -> {len(pats_after_range):,}  "
          f"lost={len(pats_before_range - pats_after_range):,}")

    bp = bp.drop_duplicates(keep="last").reset_index(drop=True)
    bp = bp.groupby(["patid", "indexdate", "eventdate"]).mean(numeric_only=True).reset_index()

    save_long_format_data(bp, save_long_format, "bp")

    print("\\n--- BP: pick closest within 1095d window ---")
    bp_best = pick_closest_within_window_before_or_at(
        bp,
        value_cols=["systolic", "diastolic"],
        window_days=WINDOW_DAYS,
        date_col="eventdate"
    )

    if not bp_best.empty:
        bp_best = bp_best.rename(columns={"eventdate": "bp_date"})
        patient = patient.merge(
            bp_best[["patid", "indexdate", "bp_date", "systolic", "diastolic"]],
            on=["patid", "indexdate"],
            how="left"
        )
    else:
        patient["bp_date"] = pd.NaT
        patient["systolic"] = np.nan
        patient["diastolic"] = np.nan

    print(f"DBG|[bp_final_on_patient] systolic missing={patient['systolic'].isna().sum():,} / {len(patient):,} "
          f"({patient['systolic'].isna().mean()*100:.2f}%)")
    print(f"DBG|[bp_final_on_patient] diastolic missing={patient['diastolic'].isna().sum():,} / {len(patient):,} "
          f"({patient['diastolic'].isna().mean()*100:.2f}%)")

    return patient
    """

    # ============================================================
    # NEW SBP-ONLY LOGIC
    # ============================================================
    bp = bp_data[["patid", "eventdate", "indexdate", "data2"]].rename(
        columns={"data2": "systolic"}
    ).copy()

    bp["eventdate"] = pd.to_datetime(bp["eventdate"], errors="coerce", dayfirst=True)
    bp = enforce_date_bounds(bp, "eventdate")
    bp["indexdate"] = parse_indexdate_iso(bp["indexdate"])

    bp["systolic"] = pd.to_numeric(bp["systolic"], errors="coerce")

    dbg(bp, "bp_raw", date_cols=["eventdate", "indexdate"])
    print(f"DBG|[bp_raw] systolic stats: {bp['systolic'].describe().to_dict()}")

    before_filter = len(bp)
    bp = bp[(bp["eventdate"].notnull()) & bp["systolic"].notnull()]
    print(f"DBG|[bp_dropna] rows: {before_filter:,} -> {len(bp):,}")

    before_range = len(bp)
    pats_before_range = set(bp['patid'].unique())
    bp = bp[(bp["systolic"] >= 40) & (bp["systolic"] <= 250)]
    pats_after_range = set(bp['patid'].unique())
    print(f"DBG|[bp_range_filter] rows: {before_range:,} -> {len(bp):,}  "
          f"patids: {len(pats_before_range):,} -> {len(pats_after_range):,}  "
          f"lost={len(pats_before_range - pats_after_range):,}")

    bp = bp.drop_duplicates(keep="last").reset_index(drop=True)
    bp = bp.groupby(["patid", "indexdate", "eventdate"]).mean(numeric_only=True).reset_index()

    save_long_format_data(bp, save_long_format, "bp")

    print("\n--- BP: pick closest SBP within window ---")
    bp_best = pick_closest_within_window_before_or_at(
        bp,
        value_cols=["systolic"],
        window_days=WINDOW_DAYS,
        date_col="eventdate"
    )

    if not bp_best.empty:
        bp_best = bp_best.rename(columns={"eventdate": "bp_date"})
        patient = patient.merge(
            bp_best[["patid", "indexdate", "bp_date", "systolic"]],
            on=["patid", "indexdate"],
            how="left"
        )
    else:
        patient["bp_date"] = pd.NaT
        patient["systolic"] = np.nan

    print(f"DBG|[bp_final_on_patient] systolic missing={patient['systolic'].isna().sum():,} / {len(patient):,} "
          f"({patient['systolic'].isna().mean()*100:.2f}%)")

    return patient
    
def load_clinical_eventdate_lookup(patient):
    print(f"\n{'='*60}")
    print("LOAD CLINICAL LOOKUP FOR ADDITIONAL LINKAGE")
    print(f"{'='*60}")

    clinical_files = sorted(glob.glob(CLINICAL_ZIP_PATTERN)) or sorted(glob.glob(CLINICAL_TXT_PATTERN))
    print(f"Found {len(clinical_files)} clinical files.")
    if not clinical_files:
        raise FileNotFoundError(f"No Clinical files found:\n  {CLINICAL_ZIP_PATTERN}\n  {CLINICAL_TXT_PATTERN}")

    cohort_patids = set(patient["patid"].astype(str).unique())
    clin_chunks = []
    total_before = 0
    total_after = 0

    for f in clinical_files:
        print(f"Processing clinical file: {f}")
        compression = "zip" if f.lower().endswith(".zip") else "infer"
        reader = pd.read_csv(f, sep="\t", dtype=str, chunksize=CHUNKSIZE, compression=compression)

        for chunk in reader:
            chunk.columns = chunk.columns.str.strip().str.lower()

            required = ["patid", "adid", "enttype", "eventdate"]
            missing = [c for c in required if c not in chunk.columns]
            if missing:
                raise KeyError(f"Clinical chunk missing columns: {missing}")

            total_before += len(chunk)

            chunk["patid"] = chunk["patid"].astype(str)
            chunk["adid"] = chunk["adid"].astype(str)
            chunk["enttype"] = chunk["enttype"].astype(str)

            chunk = chunk[chunk["patid"].isin(cohort_patids)][required].copy()
            total_after += len(chunk)

            clin_chunks.append(chunk)

    clinical_lookup = pd.concat(clin_chunks, ignore_index=True)
    clinical_lookup["eventdate"] = pd.to_datetime(clinical_lookup["eventdate"], errors="coerce", dayfirst=True)

    dbg(clinical_lookup, "clinical_lookup_pre_dedup", date_cols=["eventdate"])
    
    before_dedup = len(clinical_lookup)

    clinical_lookup["eventdate_missing"] = clinical_lookup["eventdate"].isna().astype(int)
    clinical_lookup = clinical_lookup.sort_values(
        ["patid", "adid", "enttype", "eventdate_missing", "eventdate"],
        ascending=[True, True, True, True, False]
    )
    clinical_lookup = clinical_lookup.drop_duplicates(
        subset=["patid", "adid", "enttype"],
        keep="first"
    ).reset_index(drop=True)
    clinical_lookup = clinical_lookup.drop(columns=["eventdate_missing"])

    print(f"DBG|[clinical_lookup] rows before filter={total_before:,} after cohort filter={total_after:,}")
    print(f"DBG|[clinical_lookup] rows before dedup={before_dedup:,} after dedup={len(clinical_lookup):,}")
    print(f"DBG|[clinical_lookup] missing eventdate after dedup={clinical_lookup['eventdate'].isna().sum():,}")

    return clinical_lookup
# ----------------------------------------------------------------------
# LOAD BASELINE
# ----------------------------------------------------------------------
patient = pd.read_csv(BASELINE_FILE, sep="\t", dtype=str)
patient["patid"] = patient["patid"].astype(str)
patient["indexdate"] = parse_indexdate_iso(patient["indexdate"])
dbg(patient, "baseline_loaded", date_cols=["indexdate"])
checkpoint_baseline = set(patient['patid'].unique())

clinical_lookup = load_clinical_eventdate_lookup(patient)

# ----------------------------------------------------------------------
# LOAD clinical smoking
# ----------------------------------------------------------------------
clinical_smok = pd.read_csv(CLINICAL_SMOK_FILE, sep="\t", compression="gzip", dtype=str)
clinical_smok["patid"] = clinical_smok["patid"].astype(str)
clinical_smok["medcode"] = clinical_smok["medcode"].map(norm_code)
clinical_smok["eventdate"] = pd.to_datetime(clinical_smok["eventdate"], errors="coerce", dayfirst=True)
dbg(clinical_smok, "clinical_smoking_loaded", date_cols=["eventdate"])

# ── DBG: overlap with baseline ──
smok_pats = set(clinical_smok['patid'].unique())
print(f"DBG|[clinical_smoking_overlap] baseline_patids={len(checkpoint_baseline):,}  "
      f"smoking_patids={len(smok_pats):,}  "
      f"overlap={len(checkpoint_baseline & smok_pats):,}  "
      f"baseline_without_smoking_record={len(checkpoint_baseline - smok_pats):,}")

# ----------------------------------------------------------------------
# Stream Additional files
# ----------------------------------------------------------------------
add_files = sorted(glob.glob(ADD_ZIP_PATTERN)) or sorted(glob.glob(ADD_TXT_PATTERN))
print(f"\nFound {len(add_files)} additional clinical files.")
if not add_files:
    raise FileNotFoundError(f"No Additional files found:\n  {ADD_ZIP_PATTERN}\n  {ADD_TXT_PATTERN}")

temp_bp_file = "temp_bp_data.txt"
temp_smoking_file = "temp_smoking_data.txt"
temp_weight_file = "temp_weight_data.txt"
temp_height_file = "temp_height_data.txt"

for tf in (temp_bp_file, temp_smoking_file, temp_weight_file, temp_height_file):
    if os.path.exists(tf):
        os.remove(tf)

add_row_counts = {"bp": 0, "smoking": 0, "weight": 0, "height": 0}
add_pat_counts = {"bp": set(), "smoking": set(), "weight": set(), "height": set()}

for f in add_files:
    print(f"Processing file: {f}")
    compression = "zip" if f.lower().endswith(".zip") else "infer"
    reader = pd.read_csv(f, sep="\t", dtype=str, chunksize=CHUNKSIZE, compression=compression)

    for chunk in reader:
        chunk.columns = chunk.columns.str.strip().str.lower()

        if "enttype" not in chunk.columns:
            raise KeyError("Column 'enttype' not found in Additional chunk.")
        if "patid" not in chunk.columns:
            raise KeyError("Column 'patid' not found in Additional chunk.")

        chunk["patid"] = chunk["patid"].astype(str)
        chunk["enttype"] = chunk["enttype"].astype(str)

        # keep only cohort patients BEFORE writing temp files
        before_rows = len(chunk)
        chunk = chunk[chunk["patid"].isin(checkpoint_baseline)].copy()
        after_rows = len(chunk)

        if after_rows == 0:
            continue

        print(f"DBG|[additional_chunk_filter] rows {before_rows:,} -> {after_rows:,} after cohort patid filter")

        bp_chunk = chunk[chunk["enttype"] == "1"].copy()
        if not bp_chunk.empty:
            bp_chunk.to_csv(temp_bp_file, sep="\t", mode="a", header=not os.path.exists(temp_bp_file), index=False)
            add_row_counts["bp"] += len(bp_chunk)
            add_pat_counts["bp"].update(bp_chunk["patid"].unique())

        smoking_chunk = chunk[chunk["enttype"] == "4"].copy()
        if not smoking_chunk.empty:
            smoking_chunk.to_csv(temp_smoking_file, sep="\t", mode="a", header=not os.path.exists(temp_smoking_file), index=False)
            add_row_counts["smoking"] += len(smoking_chunk)
            add_pat_counts["smoking"].update(smoking_chunk["patid"].unique())

        weight_chunk = chunk[chunk["enttype"] == "13"].copy()
        if not weight_chunk.empty:
            weight_chunk.to_csv(temp_weight_file, sep="\t", mode="a", header=not os.path.exists(temp_weight_file), index=False)
            add_row_counts["weight"] += len(weight_chunk)
            add_pat_counts["weight"].update(weight_chunk["patid"].unique())

        height_chunk = chunk[chunk["enttype"] == "14"].copy()
        if not height_chunk.empty:
            height_chunk.to_csv(temp_height_file, sep="\t", mode="a", header=not os.path.exists(temp_height_file), index=False)
            add_row_counts["height"] += len(height_chunk)
            add_pat_counts["height"].update(height_chunk["patid"].unique())

print(f"\nDBG|[additional_streaming] row counts by enttype: {add_row_counts}")
print("DBG|[additional_streaming] unique patids by enttype:")
for k in ["bp", "smoking", "weight", "height"]:
    print(f"DBG|  {k}: {len(add_pat_counts[k]):,} patids")

bp_data = pd.read_csv(temp_bp_file, sep="\t", dtype=str) if os.path.exists(temp_bp_file) else pd.DataFrame()
smoking_data = pd.read_csv(temp_smoking_file, sep="\t", dtype=str) if os.path.exists(temp_smoking_file) else pd.DataFrame()
weight_data = pd.read_csv(temp_weight_file, sep="\t", dtype=str) if os.path.exists(temp_weight_file) else pd.DataFrame()
height_data = pd.read_csv(temp_height_file, sep="\t", dtype=str) if os.path.exists(temp_height_file) else pd.DataFrame()

# ── DBG: temp file stats before Clinical linkage ──
for label, tmp_df in [("bp", bp_data), ("smoking", smoking_data), ("weight", weight_data), ("height", height_data)]:
    if not tmp_df.empty:
        dbg(tmp_df, f"additional_{label}_raw",
            date_cols=[c for c in tmp_df.columns if 'date' in c.lower()])

bp_data = attach_eventdate_from_clinical(bp_data, clinical_lookup, patient, label="bp")
smoking_data = attach_eventdate_from_clinical(smoking_data, clinical_lookup, patient, label="smoking_additional")
weight_data = attach_eventdate_from_clinical(weight_data, clinical_lookup, patient, label="weight")
height_data = attach_eventdate_from_clinical(height_data, clinical_lookup, patient, label="height")

# ── DBG: after Clinical linkage ──
for label, tmp_df in [("bp", bp_data), ("smoking", smoking_data), ("weight", weight_data), ("height", height_data)]:
    if not tmp_df.empty:
        dbg(tmp_df, f"additional_{label}_after_clinical_linkage", date_cols=["eventdate", "indexdate"])

# ----------------------------------------------------------------------
# HES
# ----------------------------------------------------------------------
hes_hosp = pd.read_csv(HES_FILE, sep="\t", dtype=str)
hes_hosp["patid"] = hes_hosp["patid"].astype(str)

if "admidate" in hes_hosp.columns:
    hes_hosp["admidate"] = pd.to_datetime(hes_hosp["admidate"], errors="coerce", dayfirst=True)
    

icd_col = "ICD" if "ICD" in hes_hosp.columns else ("diag_icd10" if "diag_icd10" in hes_hosp.columns else None)
if icd_col is None:
    raise KeyError("No ICD column found in HES file (expected 'ICD' or 'diag_icd10').")

hes_hosp["ICD"] = hes_hosp[icd_col].astype(str).str.strip().str.upper()
dbg(hes_hosp, "hes_hosp_loaded", date_cols=["admidate"] if "admidate" in hes_hosp.columns else None)

# ----------------------------------------------------------------------
# PROCESS
# ----------------------------------------------------------------------
cleaned_patient = get_smoking_data(smoking_data, clinical_smok, patient, hes_hosp)
print("get_smoking_data complete.")

wh_data = pd.concat([weight_data, height_data], ignore_index=True)
cleaned_patient = weight_height_bmi(wh_data, cleaned_patient)
print("weight_height_bmi complete.")

cleaned_patient = get_bp_data(bp_data, cleaned_patient)
print("get_bp_data complete.")

# ── DBG: final missingness summary ──
print(f"\n{'='*60}")
print("FINAL MISSINGNESS SUMMARY")
print(f"{'='*60}")
dbg(cleaned_patient, "final_output", date_cols=["indexdate"])
final_pats = set(cleaned_patient['patid'].unique())
dbg_patid_diff(checkpoint_baseline, final_pats, "baseline_vs_final")

#for col in ["smoking_status", "bmi", "systolic", "diastolic"]:
for col in ["smoking_status", "bmi", "systolic"]:
    if col in cleaned_patient.columns:
        n_miss = cleaned_patient[col].isna().sum()
        pct = n_miss / len(cleaned_patient) * 100
        print(f"DBG|[final_missing] {col}: {n_miss:,} / {len(cleaned_patient):,} ({pct:.2f}%)")

# ----------------------------------------------------------------------
# SAVE
# ----------------------------------------------------------------------
cleaned_patient.to_csv(OUTPUT_FILE, sep="\t", index=False)
print(f"\nCleaned data saved to {OUTPUT_FILE}")
