#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOLD Script 4 (CORRECTED): Baseline enrichment with demographics + CPRD ethnicity fallback
  – Matches AURUM logic: grouped baseline, HES ethnicity primary + CPRD fallback,
    simple death flag (dod <= STUDY_END), gender decoded, pracid included.
"""

import os
import zipfile
from glob import glob
from datetime import datetime
import time

import pandas as pd
import numpy as np


# =============================================================================
# Debug helpers (matching AURUM style)
# =============================================================================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def dbg(tag: str, msg: str) -> None:
    print(f"DBG| [{tag}] {msg}", flush=True)


def dbg_date_col(df: pd.DataFrame, col: str, tag: str) -> None:
    s = pd.to_datetime(df[col], errors="coerce")
    missing = s.isna().sum()
    valid = s.dropna()
    if len(valid) == 0:
        dbg(tag, f"  - {col}: missing={missing} (ALL missing)")
        return
    years = valid.dt.year
    dbg(tag, f"  - {col}: missing={missing}")
    dbg(tag, f"    year: min={int(years.min())} p50={int(years.median())} max={int(years.max())} "
             f">2025={int((years > 2025).sum())} <1900={int((years < 1900).sum())} "
             f"==9999={int((years == 9999).sum())}")


def dbg_df(df: pd.DataFrame, tag: str, patid_col: str = "patid") -> None:
    n_patids = df[patid_col].nunique() if patid_col in df.columns else "N/A"
    dbg(tag, f"rows={len(df):,}  patids={n_patids:,}")


# =============================================================================
# Paths
# =============================================================================
chunk_dir = "/scratch/alice/b/bg205/28_02_GOLD"

baseline_path = os.path.join(chunk_dir, "gold_baseline_grouped_df_NoNA.txt")

patient_zip_path = (
    "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/"
    "FZ_GOLD_All_Extract_Patient_001.zip"
)

hes_eth_path = (
    "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/"
    "GOLD_linked/hes_patient_23_002869_DM.txt"
)

ethnicity_codes_path = "/scratch/alice/b/bg205/GOLD_Codes_FZ.xlsx"

clinical_zips_dir = "/scratch/alice/b/bg205/DataCleaning_Gold_v2/GOLD/"
clinical_zips_glob = os.path.join(clinical_zips_dir, "FZ_GOLD_All_Extract_Clinical_*.zip")

imd_path = (
    "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/"
    "GOLD_linked/patient_2019_imd_23_002869.txt"
)

death_path = (
    "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/"
    "GOLD_linked/death_patient_23_002869_DM.txt"
)

output_path = os.path.join(chunk_dir, "Enriched_baseline_with_demographics.txt")

STUDY_END = pd.Timestamp("2021-03-31")


# =============================================================================
# Helper: read .txt from .zip
# =============================================================================
def read_txt_from_zip(zip_path, **read_csv_kwargs):
    with zipfile.ZipFile(zip_path) as zf:
        txt_members = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        if not txt_members:
            raise FileNotFoundError(f"No .txt found inside {zip_path}")
        with zf.open(txt_members[0]) as fh:
            return pd.read_csv(fh, sep="\t", **read_csv_kwargs)


# =============================================================================
# 0) Load baseline (GROUPED — one row per patient, matches AURUM)
# =============================================================================
t0 = time.time()
log(f"Loading baseline: {baseline_path}")
baseline_df = pd.read_csv(baseline_path, sep="\t", dtype={"patid": str})
log(f"Baseline loaded: shape={baseline_df.shape}")

dbg_df(baseline_df, "BASELINE_LOADED")
dbg("BASELINE_LOADED", f"columns={list(baseline_df.columns)}")
dbg("BASELINE_LOADED", f"dtypes:\n{baseline_df.dtypes.to_string()}")
dbg("BASELINE_LOADED", f"sample patids: {baseline_df['patid'].head(5).tolist()}")

for c in baseline_df.columns:
    if "date" in c.lower() or c.lower() in ("obsdate", "indexdate", "eventdate"):
        dbg_date_col(baseline_df, c, "BASELINE_LOADED")

baseline_patids = set(baseline_df["patid"].astype(str))
log(f"Baseline patids in set: n={len(baseline_patids):,}")


# =============================================================================
# 1) Patient: gender, yob, pracid (FILTERED TO BASELINE — matches AURUM)
# =============================================================================
log("Loading patient table from ZIP")
df_patient = read_txt_from_zip(
    patient_zip_path,
    dtype=str,
    low_memory=False
)
df_patient.columns = [c.lower() for c in df_patient.columns]

# Keep only what we need
# Keep only what we need
want_cols = [c for c in ["patid", "gender", "yob"] if c in df_patient.columns]
df_patient = df_patient[want_cols].copy()
df_patient["patid"] = df_patient["patid"].astype(str)

# Derive pracid from last 5 digits of patid (per CPRD GOLD specification)
df_patient["pracid"] = df_patient["patid"].str[-5:]

dbg_df(df_patient, "PATIENT_RAW")
dbg("PATIENT_RAW", f"columns={list(df_patient.columns)}")
dbg("PATIENT_RAW", f"gender value_counts:\n{df_patient['gender'].value_counts(dropna=False).to_string()}")
yob_num = pd.to_numeric(df_patient['yob'], errors='coerce')
dbg("PATIENT_RAW", f"yob: min={yob_num.min():.0f}  max={yob_num.max():.0f}  "
                    f"median={yob_num.median():.0f}  missing={df_patient['yob'].isna().sum()}")

# Filter to baseline
raw_count = len(df_patient)
df_patient = df_patient[df_patient["patid"].isin(baseline_patids)]
df_patient = df_patient.drop_duplicates(subset=["patid"])

dbg("PATIENT_FILTERED", f"raw={raw_count:,}  filtered={len(df_patient):,}")
dbg_df(df_patient, "PATIENT_FINAL")

missing_from_patient = baseline_patids - set(df_patient["patid"])
dbg("PATIENT_COVERAGE", f"baseline={len(baseline_patids):,}  found={len(df_patient):,}  "
                         f"missing={len(missing_from_patient):,}")


# =============================================================================
# 2) Decode gender (matches AURUM — GOLD uses standard CPRD codes)
# =============================================================================
log("Decoding gender")
gender_map = {"1": "Male", "2": "Female", "3": "Indeterminate", "0": "Data Not Entered"}

df_patient["gender"] = df_patient["gender"].astype(str).str.strip().map(gender_map)

log(f"Gender decoded. Missing: {df_patient['gender'].isna().sum():,}")
dbg("GENDER_DECODED", f"value_counts:\n{df_patient['gender'].value_counts(dropna=False).to_string()}")


# =============================================================================
# 3a) HES ethnicity (primary — matches AURUM)
# =============================================================================
log("Loading HES ethnicity (primary)")
hes_eth = pd.read_csv(
    hes_eth_path,
    sep="\t",
    usecols=["patid", "gen_ethnicity"],
    dtype={"patid": str}
).drop_duplicates("patid")

dbg("HES_ETH_RAW", f"rows after dedup={len(hes_eth):,}")

hes_eth = hes_eth[hes_eth["patid"].isin(baseline_patids)]
log(f"HES ethnicity: shape={hes_eth.shape}")
dbg_df(hes_eth, "HES_ETH_FILTERED")
dbg("HES_ETH_FILTERED", f"gen_ethnicity value_counts:\n"
    f"{hes_eth['gen_ethnicity'].value_counts(dropna=False).to_string()}")
dbg("HES_ETH_FILTERED", f"coverage: {len(hes_eth):,}/{len(baseline_patids):,} = "
                          f"{100*len(hes_eth)/len(baseline_patids):.2f}%")


# =============================================================================
# 3b) Build medcode -> ethnicity map from GOLD Excel (matches AURUM structure)
# =============================================================================
log(f"Building ethnicity medcode map from {ethnicity_codes_path}")
eth_codes = pd.read_excel(ethnicity_codes_path, sheet_name="Ethn", dtype=str)

dbg("ETH_EXCEL_RAW", f"columns={list(eth_codes.columns)}  rows={len(eth_codes)}")
dbg("ETH_EXCEL_RAW", f"sample:\n{eth_codes.head()}")

# Identify the medcode and ethnicity group columns
medcode_col = None
for candidate in ["medcode", "Medcode", "MedCode", "medcodeid", "code"]:
    if candidate in eth_codes.columns:
        medcode_col = candidate
        break
if medcode_col is None:
    raise KeyError(f"Cannot find medcode column in ethnicity Excel. Available: {list(eth_codes.columns)}")

eth_group_col = None
for candidate in ["ethnic", "ethnicity", "Ethnic", "Ethnicity", "gen_ethnicity", "group"]:
    if candidate in eth_codes.columns:
        eth_group_col = candidate
        break
if eth_group_col is None:
    raise KeyError(f"Cannot find ethnicity group column in Excel. Available: {list(eth_codes.columns)}")

dbg("ETH_EXCEL", f"using medcode_col='{medcode_col}'  eth_group_col='{eth_group_col}'")

eth_codes[medcode_col] = eth_codes[medcode_col].astype(str).str.strip()
eth_codes = eth_codes.dropna(subset=[medcode_col, eth_group_col])

# Harmonise group names to match AURUM output
# AURUM groups: Black, Missing, Other Mixed, South Asian, White
# Supervisor GOLD groups: White, Black, South Asian, Mixed/Other, Unknown
group_remap = {
    "Mixed/Other": "Other Mixed",
    "Unknown": "Missing",
}
eth_codes["gen_ethnicity"] = eth_codes[eth_group_col].replace(group_remap)

dbg("ETH_EXCEL", f"group breakdown after remap:\n"
    f"{eth_codes['gen_ethnicity'].value_counts().to_string()}")

ethnicity_set = set(eth_codes[medcode_col])
ethnicity_dict = dict(zip(eth_codes[medcode_col], eth_codes["gen_ethnicity"]))
log(f"Ethnicity medcodes loaded: n={len(ethnicity_set):,}")

# Check for duplicates (same medcode -> multiple groups)
dupes = eth_codes.groupby(medcode_col)["gen_ethnicity"].nunique()
dupes = dupes[dupes > 1]
if len(dupes) > 0:
    dbg("ETH_MAP_WARNING", f"medcodes mapping to MULTIPLE groups: {len(dupes)}")
    dbg("ETH_MAP_WARNING", f"examples:\n{dupes.head(10).to_string()}")
else:
    dbg("ETH_MAP", "no duplicate medcode->group mappings (good)")


# =============================================================================
# 3c) CPRD ethnicity fallback from RAW Clinical ZIPs (STREAMING — matches AURUM)
#     Earliest ethnicity record per patient
# =============================================================================
clinical_zips = sorted(glob(clinical_zips_glob))
if not clinical_zips:
    raise FileNotFoundError(f"No Clinical ZIPs found: {clinical_zips_glob}")

log(f"RAW Clinical ZIPs found: n={len(clinical_zips)}")
dbg("CLIN_ZIPS", f"first={os.path.basename(clinical_zips[0])}  "
                  f"last={os.path.basename(clinical_zips[-1])}")
log("Starting Clinical ethnicity fallback loop...")

best = {}  # patid -> (eventdate, gen_ethnicity)
chunksize = 200_000

total_chunks_read = 0
total_rows_read = 0
total_eth_matches = 0
total_baseline_matches = 0


def should_replace(existing_dt, new_dt):
    """Pick the earlier date (matches AURUM logic)."""
    if pd.isna(existing_dt) and pd.notna(new_dt):
        return True
    if pd.notna(existing_dt) and pd.isna(new_dt):
        return False
    if pd.isna(existing_dt) and pd.isna(new_dt):
        return False
    return new_dt < existing_dt


loop_start = time.time()

for i, zpath in enumerate(clinical_zips, start=1):
    zip_start = time.time()

    with zipfile.ZipFile(zpath) as z:
        txt_members = [m for m in z.namelist() if m.lower().endswith(".txt")]
        if not txt_members:
            dbg("CLIN_ZIP_SKIP", f"no .txt in {os.path.basename(zpath)}")
            continue

        for member in txt_members:
            with z.open(member) as f:
                reader = pd.read_csv(
                    f,
                    sep="\t",
                    dtype=str,
                    usecols=["patid", "medcode", "eventdate"],
                    chunksize=chunksize
                )

                for chunk in reader:
                    total_chunks_read += 1
                    total_rows_read += len(chunk)

                    chunk["medcode"] = chunk["medcode"].astype(str).str.strip()

                    # Filter: ethnicity medcodes first (small set), then baseline
                    chunk = chunk[chunk["medcode"].isin(ethnicity_set)]
                    if chunk.empty:
                        continue

                    total_eth_matches += len(chunk)

                    chunk["patid"] = chunk["patid"].astype(str)
                    chunk = chunk[chunk["patid"].isin(baseline_patids)]
                    if chunk.empty:
                        continue

                    total_baseline_matches += len(chunk)

                    chunk["gen_ethnicity"] = chunk["medcode"].map(ethnicity_dict)
                    chunk["eventdate"] = pd.to_datetime(
                        chunk["eventdate"], errors="coerce", dayfirst=True
                    )

                    # Earliest per patid in this chunk
                    min_dt = chunk.groupby("patid", sort=False)["eventdate"].min()
                    chunk = chunk.merge(min_dt.rename("min_eventdate"), on="patid", how="left")
                    chunk = chunk[chunk["eventdate"] == chunk["min_eventdate"]]
                    chunk = chunk.drop_duplicates("patid", keep="first")[
                        ["patid", "eventdate", "gen_ethnicity"]
                    ]

                    for row in chunk.itertuples(index=False):
                        pat = row.patid
                        dt = row.eventdate
                        eth = row.gen_ethnicity
                        if pat not in best:
                            best[pat] = (dt, eth)
                        else:
                            if should_replace(best[pat][0], dt):
                                best[pat] = (dt, eth)

    if i % 5 == 0:
        elapsed = time.time() - loop_start
        rate = i / elapsed if elapsed > 0 else 0
        remaining = (len(clinical_zips) - i) / rate if rate > 0 else float("inf")
        log(f"Processed {i}/{len(clinical_zips)} clinical ZIPs | "
            f"last_zip={time.time()-zip_start:.1f}s | "
            f"ETA~{remaining/3600:.2f}h | best_size={len(best):,}")
        dbg("CLIN_PROGRESS", f"total_chunks={total_chunks_read:,}  total_rows={total_rows_read:,}  "
                              f"eth_matches={total_eth_matches:,}  baseline_matches={total_baseline_matches:,}")

log(f"Finished clinical ethnicity loop. best_size={len(best):,}")
dbg("CLIN_ETH_DONE", f"total_chunks={total_chunks_read:,}  total_rows_read={total_rows_read:,}")
dbg("CLIN_ETH_DONE", f"total_eth_matches={total_eth_matches:,}  "
                       f"total_baseline_matches={total_baseline_matches:,}")
dbg("CLIN_ETH_DONE", f"unique patients with CPRD ethnicity={len(best):,}")

if best:
    from collections import Counter
    cprd_eth_counts = Counter(v[1] for v in best.values())
    dbg("CPRD_ETH_DIST", f"group breakdown:\n" +
        "\n".join(f"  {k}: {v:,}" for k, v in sorted(cprd_eth_counts.items(), key=lambda x: -x[1])))

obs_eth = pd.DataFrame(
    [(p, v[1]) for p, v in best.items()],
    columns=["patid", "gen_ethnicity"]
)
log(f"CPRD ethnicity fallback rows: n={obs_eth.shape[0]:,}")


# =============================================================================
# 3d) Combine HES + CPRD ethnicity (HES priority — matches AURUM)
# =============================================================================
eth_combined = (
    hes_eth.set_index("patid")
    .combine_first(obs_eth.set_index("patid"))
    .reset_index()
)
log(f"Ethnicity combined (HES priority) rows: n={eth_combined.shape[0]:,}")
dbg("ETH_COMBINED", f"gen_ethnicity value_counts:\n"
    f"{eth_combined['gen_ethnicity'].value_counts(dropna=False).to_string()}")

hes_patids = set(hes_eth["patid"].astype(str))
cprd_patids = set(obs_eth["patid"].astype(str))
has_eth = eth_combined[eth_combined["gen_ethnicity"].notna()]
still_na = baseline_patids - set(has_eth["patid"].astype(str))
dbg("ETH_SOURCE", f"HES_only={len(hes_patids - cprd_patids):,}  "
                    f"CPRD_only={len(cprd_patids - hes_patids):,}  "
                    f"both={len(hes_patids & cprd_patids):,}")
dbg("ETH_SOURCE", f"still_NA_ethnicity={len(still_na):,}")
dbg("ETH_SOURCE", f"total coverage: {len(baseline_patids)-len(still_na):,}/{len(baseline_patids):,} = "
                   f"{100*(len(baseline_patids)-len(still_na))/len(baseline_patids):.2f}%")


# =============================================================================
# 4) IMD (matches AURUM)
# =============================================================================
log("Loading IMD")
imd = pd.read_csv(
    imd_path,
    sep="\t",
    usecols=["patid", "e2019_imd_10"],
    dtype={"patid": str}
).drop_duplicates("patid")

dbg("IMD_RAW", f"rows after dedup={len(imd):,}")

imd = imd[imd["patid"].isin(baseline_patids)]
log(f"IMD: shape={imd.shape}")
dbg_df(imd, "IMD_FILTERED")
dbg("IMD_FILTERED", f"e2019_imd_10 value_counts:\n"
    f"{imd['e2019_imd_10'].value_counts(dropna=False).sort_index().to_string()}")
dbg("IMD_FILTERED", f"coverage: {len(imd):,}/{len(baseline_patids):,} = "
                     f"{100*len(imd)/len(baseline_patids):.2f}%")


# =============================================================================
# 5) Death — simple flag: dod <= STUDY_END (matches AURUM, NO indexdate condition)
# =============================================================================
log("Loading ONS death table")
death_ons = pd.read_csv(
    death_path,
    sep="\t",
    usecols=["patid", "dod"],
    dtype={"patid": str}
).drop_duplicates("patid")

dbg("DEATH_RAW", f"rows after dedup={len(death_ons):,}")

death_ons = death_ons[death_ons["patid"].isin(baseline_patids)]
death_ons = death_ons.rename(columns={"dod": "dod_ons"})
death_ons["dod_ons"] = pd.to_datetime(death_ons["dod_ons"], errors="coerce", dayfirst=True)

log(f"ONS death table: shape={death_ons.shape}")
log(f"ONS deaths with valid date: {death_ons['dod_ons'].notna().sum():,}")
dbg_df(death_ons, "DEATH_FILTERED")
dbg_date_col(death_ons, "dod_ons", "DEATH_FILTERED")
dbg("DEATH_FILTERED", f"coverage: {len(death_ons):,}/{len(baseline_patids):,} = "
                       f"{100*len(death_ons)/len(baseline_patids):.2f}%")

# Death flag: died AND dod <= study end (matches AURUM — no indexdate condition)
death_ons["death_ons"] = (
    (death_ons["dod_ons"].notna()) & (death_ons["dod_ons"] <= STUDY_END)
).astype(int)

# Censor date
death_ons["censor_date"] = death_ons["dod_ons"]
death_ons.loc[
    death_ons["censor_date"].isna() | (death_ons["censor_date"] > STUDY_END),
    "censor_date"
] = STUDY_END

dbg("DEATH_OUTCOME", f"death_ons=1: {(death_ons['death_ons']==1).sum():,}  "
                      f"death_ons=0: {(death_ons['death_ons']==0).sum():,}")
dbg("DEATH_OUTCOME", f"deaths AFTER study_end (censored): "
                      f"{(death_ons['dod_ons'].notna() & (death_ons['dod_ons'] > STUDY_END)).sum():,}")
dbg_date_col(death_ons, "censor_date", "DEATH_CENSOR")


# =============================================================================
# 6) Merge demographics (matches AURUM merge order)
# =============================================================================
log("Merging demographics")
dbg("PRE_DEMO_MERGE", f"patient={len(df_patient):,}  eth_combined={len(eth_combined):,}  "
                       f"imd={len(imd):,}  death_ons={len(death_ons):,}")

demographics = (
    df_patient
    .merge(eth_combined, on="patid", how="left")
    .merge(imd, on="patid", how="left")
    .merge(death_ons[["patid", "dod_ons", "death_ons", "censor_date"]], on="patid", how="left")
)
log(f"Demographics merged: shape={demographics.shape}")
dbg_df(demographics, "DEMOGRAPHICS")
dbg("DEMOGRAPHICS", f"columns={list(demographics.columns)}")

for col in demographics.columns:
    n_miss = demographics[col].isna().sum()
    if n_miss > 0:
        dbg("DEMOGRAPHICS_MISSING", f"{col}: {n_miss:,} ({100*n_miss/len(demographics):.2f}%)")


# =============================================================================
# 7) Enrich baseline and save (matches AURUM)
# =============================================================================
log("Merging demographics onto baseline")
pre_enrich_rows = len(baseline_df)
enriched = baseline_df.merge(demographics, on="patid", how="left")

dbg("ENRICH_MERGE", f"baseline_rows={pre_enrich_rows:,}  enriched_rows={len(enriched):,}  "
                     f"(should be equal for left join)")
if len(enriched) != pre_enrich_rows:
    dbg("ENRICH_MERGE_WARNING", f"ROW COUNT CHANGED! diff={len(enriched)-pre_enrich_rows:+,}")

enriched["death_ons"] = enriched["death_ons"].fillna(0).astype(int)
enriched["censor_date"] = pd.to_datetime(enriched["censor_date"], errors="coerce")
enriched.loc[enriched["censor_date"].isna(), "censor_date"] = STUDY_END

log(f"Enriched baseline: shape={enriched.shape}")
log(f"Missing censor_date after fill: {enriched['censor_date'].isna().sum():,}")

dbg_df(enriched, "ENRICHED_FINAL")
dbg("ENRICHED_FINAL", f"columns={list(enriched.columns)}")

# Final missingness report
dbg("ENRICHED_FINAL", "--- MISSINGNESS REPORT ---")
for col in enriched.columns:
    n_miss = enriched[col].isna().sum()
    pct = 100 * n_miss / len(enriched)
    dbg("ENRICHED_FINAL", f"  {col}: missing={n_miss:,} ({pct:.2f}%)")

# Value counts for key categorical columns
for col in ["gender", "gen_ethnicity", "e2019_imd_10", "death_ons"]:
    if col in enriched.columns:
        dbg("ENRICHED_FINAL", f"{col} value_counts:\n"
            f"{enriched[col].value_counts(dropna=False).to_string()}")

# Date diagnostics on final output
for col in enriched.columns:
    if "date" in col.lower() or col.lower() in ("indexdate", "dod_ons"):
        dbg_date_col(enriched, col, "ENRICHED_FINAL")

if "diabetes_type" in enriched.columns:
    dbg("ENRICHED_FINAL", f"diabetes_type distribution:\n"
        f"{enriched['diabetes_type'].value_counts(dropna=False).to_string()}")

log(f"Writing output: {output_path}")
enriched.to_csv(output_path, sep="\t", index=False)
log("Saved successfully.")

dbg("OUTPUT", f"file={output_path}  size_bytes={os.path.getsize(output_path):,}")
log(f"TOTAL runtime: {(time.time() - t0)/3600:.2f} hours")
