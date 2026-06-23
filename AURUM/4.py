#!/usr/bin/env python3
"""
AURUM baseline enrichment (demographics + ethnicity fallback from RAW Observation ZIPs)

Key improvements vs your current version:
- Unbuffered-friendly logging with timestamps + flush (so SLURM .out always updates)
- Patient step filters to baseline_patids (huge speed + RAM win)
- Observation filtering order optimized (ethnicity medcodes first, then baseline patids)
- Fewer expensive sorts: uses groupby(min) per chunk rather than sort_values
- More frequent progress + ETA
- Safer gender decode merge + column handling
"""

import os
import zipfile
from glob import glob
from datetime import datetime
import time

import pandas as pd


# -----------------------
# Logging
# -----------------------
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def dbg(tag: str, msg: str) -> None:
    print(f"DBG| [{tag}] {msg}", flush=True)


def dbg_date_col(df: pd.DataFrame, col: str, tag: str) -> None:
    """Print date column diagnostics matching script 3 style."""
    s = pd.to_datetime(df[col], errors="coerce")
    missing = s.isna().sum()
    valid = s.dropna()
    if len(valid) == 0:
        dbg(tag, f"  - {col}: missing={missing} (ALL missing, no valid dates)")
        return
    years = valid.dt.year
    dbg(tag, f"  - {col}: missing={missing}")
    dbg(tag, f"    year: min={int(years.min())} p50={int(years.median())} max={int(years.max())} "
             f">2025={int((years > 2025).sum())} <1900={int((years < 1900).sum())} ==9999={int((years == 9999).sum())}")


def dbg_df(df: pd.DataFrame, tag: str, patid_col: str = "patid") -> None:
    """Print shape + unique patids."""
    n_patids = df[patid_col].nunique() if patid_col in df.columns else "N/A"
    dbg(tag, f"rows={len(df):,}  patids={n_patids:,}")


# -----------------------
# Paths
# -----------------------
patient_dir = "/scratch/alice/b/bg205/DataCleaning_Aurum_v2/input_patient"
obs_dir = "/scratch/alice/b/bg205/DataCleaning_Aurum_v2/Observation/"
gender_lookup_zip = "/scratch/alice/b/bg205/DataCleaning_Aurum_v2/lookups/202205_Lookups_CPRDAurum.zip"
hes_eth_path = "/scratch/alice/b/bg205/DataCleaning_Aurum_v2/linkage/hes_patient_23_002869_DM.txt"
imd_path = "/scratch/alice/b/bg205/DataCleaning_Aurum_v2/linkage/patient_2019_imd_23_002869.txt"
death_path = "/scratch/alice/b/bg205/DataCleaning_Aurum_v2/linkage/death_patient_23_002869_DM.txt"
baseline_path = "/scratch/alice/b/bg205/01_03_AURUM/aurum_baseline_grouped_df_NoNA.txt"
codes_dir = "/scratch/alice/b/bg205/DataCleaning_Aurum_v2/ethnicity_codes"

out_dir = "/scratch/alice/b/bg205/01_03_AURUM"
output_path = os.path.join(out_dir, "Enriched_baseline_with_demographics.txt")

STUDY_END = pd.Timestamp("2021-03-31")  # 3 year lookback (not used in filtering, kept for reference)


# -----------------------
# 0) Load baseline
# -----------------------
t0 = time.time()
log(f"Loading baseline: {baseline_path}")
baseline_df = pd.read_csv(baseline_path, sep="\t", dtype={"patid": str})
log(f"Baseline loaded: shape={baseline_df.shape}")

dbg_df(baseline_df, "BASELINE_LOADED")
dbg("BASELINE_LOADED", f"columns={list(baseline_df.columns)}")
dbg("BASELINE_LOADED", f"dtypes:\n{baseline_df.dtypes.to_string()}")
dbg("BASELINE_LOADED", f"sample patids: {baseline_df['patid'].head(5).tolist()}")

# check for any date columns already present
for c in baseline_df.columns:
    if "date" in c.lower() or c.lower() in ("obsdate", "indexdate"):
        dbg_date_col(baseline_df, c, "BASELINE_LOADED")

# speed filter for all downstream joins/filters
baseline_patids = set(baseline_df["patid"].astype(str))
log(f"Baseline patids in set: n={len(baseline_patids):,}")


# -----------------------
# 1) Patient: gender, yob, pracid (FILTERED TO BASELINE)
# -----------------------
zipped_files = sorted(glob(os.path.join(patient_dir, "*.zip")))
if not zipped_files:
    raise FileNotFoundError(f"No patient ZIPs found in {patient_dir}")
log(f"Patient ZIPs found: n={len(zipped_files)}")
dbg("PATIENT_ZIPS", f"first={zipped_files[0]}  last={zipped_files[-1]}")

want = {"patid", "gender", "yob", "pracid", "practiceid"}
patient_dfs = []
total_raw_patient_rows = 0
total_filtered_patient_rows = 0

for idx, zf in enumerate(zipped_files, start=1):
    with zipfile.ZipFile(zf) as z:
        txt_file = next((f for f in z.namelist() if f.lower().endswith(".txt")), None)
        if txt_file is None:
            dbg("PATIENT_ZIP_SKIP", f"No .txt in {os.path.basename(zf)}")
            continue
        with z.open(txt_file) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                dtype=str,
                usecols=lambda c: c.lower() in want
            )
    df.columns = [c.lower() for c in df.columns]
    if "pracid" not in df.columns and "practiceid" in df.columns:
        df = df.rename(columns={"practiceid": "pracid"})

    raw_rows = len(df)
    total_raw_patient_rows += raw_rows

    # crucial: restrict to cohort early
    df["patid"] = df["patid"].astype(str)
    df = df[df["patid"].isin(baseline_patids)]

    filtered_rows = len(df)
    total_filtered_patient_rows += filtered_rows

    if df.empty:
        continue

    patient_dfs.append(df)

    if idx % 5 == 0:
        log(f"Loaded patient zips: {idx}/{len(zipped_files)}")
        dbg("PATIENT_PROGRESS", f"raw_rows_so_far={total_raw_patient_rows:,}  filtered_rows_so_far={total_filtered_patient_rows:,}")

if not patient_dfs:
    raise RuntimeError("Patient step produced no rows after filtering to baseline_patids.")

dbg("PATIENT_RAW_TOTAL", f"raw_rows={total_raw_patient_rows:,}  filtered_rows={total_filtered_patient_rows:,}  "
                          f"pct_kept={100*total_filtered_patient_rows/max(total_raw_patient_rows,1):.2f}%")

patient_all = pd.concat(patient_dfs, ignore_index=True)
dbg("PATIENT_PRE_DEDUP", f"rows={len(patient_all):,}  unique_patids={patient_all['patid'].nunique():,}")

patient_all = patient_all.drop_duplicates(subset=["patid"])
log(f"Patient table built: shape={patient_all.shape}")
dbg_df(patient_all, "PATIENT_FINAL")
dbg("PATIENT_FINAL", f"columns={list(patient_all.columns)}")
dbg("PATIENT_FINAL", f"gender value_counts:\n{patient_all['gender'].value_counts(dropna=False).to_string()}")
dbg("PATIENT_FINAL", f"yob: min={patient_all['yob'].astype(float).min():.0f}  "
                      f"max={patient_all['yob'].astype(float).max():.0f}  "
                      f"median={patient_all['yob'].astype(float).median():.0f}  "
                      f"missing={patient_all['yob'].isna().sum()}")

# Check: how many baseline patients are NOT in patient table?
patient_patids = set(patient_all["patid"].astype(str))
missing_from_patient = baseline_patids - patient_patids
dbg("PATIENT_COVERAGE", f"baseline_patids={len(baseline_patids):,}  found_in_patient={len(patient_patids):,}  "
                         f"missing={len(missing_from_patient):,}")


# -----------------------
# 2) Decode gender (lookup)
# -----------------------
log("Decoding gender via lookup")
with zipfile.ZipFile(gender_lookup_zip) as z:
    gender_file = next((f for f in z.namelist() if f.lower().endswith("gender.txt")), None)
    if gender_file is None:
        raise FileNotFoundError("Could not find gender.txt inside gender_lookup_zip")
    with z.open(gender_file) as f:
        gender_map = pd.read_csv(f, sep="\t", dtype=str)

gender_map.columns = [c.strip() for c in gender_map.columns]
dbg("GENDER_LOOKUP", f"raw columns={list(gender_map.columns)}  rows={len(gender_map)}")
dbg("GENDER_LOOKUP", f"contents:\n{gender_map.to_string()}")

# Try to find expected columns; fall back safely
gender_id_col = "genderid" if "genderid" in [c.lower() for c in gender_map.columns] else None
# Normalize to lower-case for matching
gender_map.columns = [c.lower() for c in gender_map.columns]
desc_col = "description" if "description" in gender_map.columns else gender_map.columns[-1]
if gender_id_col is None:
    gender_id_col = "genderid" if "genderid" in gender_map.columns else gender_map.columns[0]

dbg("GENDER_LOOKUP", f"using gender_id_col='{gender_id_col}'  desc_col='{desc_col}'")

pre_merge_rows = len(patient_all)
patient_all = patient_all.merge(
    gender_map[[gender_id_col, desc_col]],
    left_on="gender",
    right_on=gender_id_col,
    how="left"
)
dbg("GENDER_MERGE", f"rows before={pre_merge_rows:,}  after={len(patient_all):,}  "
                     f"(should be equal unless lookup has dupes)")

patient_all = (
    patient_all
    .drop(columns=[gender_id_col], errors="ignore")
    .drop(columns=["gender"], errors="ignore")
    .rename(columns={desc_col: "gender"})
)

log(f"Gender decoded. Missing gender: {patient_all['gender'].isna().sum():,}")
dbg("GENDER_DECODED", f"gender value_counts:\n{patient_all['gender'].value_counts(dropna=False).to_string()}")


# -----------------------
# 3a) HES ethnicity (primary)
# -----------------------
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
dbg("HES_ETH_FILTERED", f"gen_ethnicity value_counts:\n{hes_eth['gen_ethnicity'].value_counts(dropna=False).to_string()}")
dbg("HES_ETH_FILTERED", f"coverage: {len(hes_eth):,}/{len(baseline_patids):,} = "
                          f"{100*len(hes_eth)/len(baseline_patids):.2f}%")


# -----------------------
# 3b) Build medcodeid -> ethnicity map from CSVs
# -----------------------
log("Building ethnicity medcode map")
wanted_groups = {"Black", "Missing", "Other_Mixed", "South_Asian", "White"}
eth_map_list = []

for fp in sorted(glob(os.path.join(codes_dir, "*.csv"))):
    name = os.path.splitext(os.path.basename(fp))[0]
    if name not in wanted_groups:
        dbg("ETH_CSV_SKIP", f"skipping {name} (not in wanted_groups)")
        continue
    df = pd.read_csv(fp, header=2, usecols=["medcodeid"], dtype={"medcodeid": str})
    df["medcodeid"] = df["medcodeid"].astype(str).str.strip()
    df["gen_ethnicity"] = name.replace("_", " ")
    dbg("ETH_CSV_LOADED", f"{name}: {len(df)} medcodes")
    eth_map_list.append(df)

if not eth_map_list:
    raise RuntimeError("No ethnicity CSVs loaded — check codes_dir")

ethnicity_map = pd.concat(eth_map_list, ignore_index=True).dropna()
ethnicity_map["medcodeid"] = ethnicity_map["medcodeid"].astype(str).str.strip()

ethnicity_set = set(ethnicity_map["medcodeid"].astype(str))
ethnicity_dict = dict(zip(ethnicity_map["medcodeid"].astype(str), ethnicity_map["gen_ethnicity"]))
log(f"Ethnicity medcodes loaded: n={len(ethnicity_set):,}")
dbg("ETH_MAP", f"total unique medcodes={len(ethnicity_set):,}")
dbg("ETH_MAP", f"group breakdown:\n{ethnicity_map['gen_ethnicity'].value_counts().to_string()}")

# check for duplicates (same medcode -> multiple groups)
dupes = ethnicity_map.groupby("medcodeid")["gen_ethnicity"].nunique()
dupes = dupes[dupes > 1]
if len(dupes) > 0:
    dbg("ETH_MAP_WARNING", f"medcodes mapping to MULTIPLE groups: {len(dupes)}")
    dbg("ETH_MAP_WARNING", f"examples: {dupes.head(10).to_string()}")
else:
    dbg("ETH_MAP", "no duplicate medcode->group mappings (good)")


# -----------------------
# 3c/3d) CPRD ethnicity fallback from RAW Observation ZIPs (STREAMING)
# Earliest ethnicity record per patient
# -----------------------
obs_zips = sorted(glob(os.path.join(obs_dir, "FZ_Aurum_*_Extract_Observation_*.zip")))
if not obs_zips:
    raise FileNotFoundError(
        f"No RAW Observation ZIPs found in {obs_dir} with pattern FZ_Aurum_*_Extract_Observation_*.zip"
    )
log(f"RAW Observation ZIPs found: n={len(obs_zips)}")
dbg("OBS_ZIPS", f"first={os.path.basename(obs_zips[0])}  last={os.path.basename(obs_zips[-1])}")
log("Starting Observation ethnicity fallback loop...")

best = {}  # patid -> (obsdate, gen_ethnicity)
usecols = ["patid", "medcodeid", "obsdate"]
chunksize = 200_000

total_chunks_read = 0
total_rows_read = 0
total_eth_matches = 0
total_baseline_matches = 0

def should_replace(existing_dt, new_dt):
    if pd.isna(existing_dt) and pd.notna(new_dt):
        return True
    if pd.notna(existing_dt) and pd.isna(new_dt):
        return False
    if pd.isna(existing_dt) and pd.isna(new_dt):
        return False
    return new_dt < existing_dt

loop_start = time.time()

for i, zpath in enumerate(obs_zips, start=1):
    zip_start = time.time()

    with zipfile.ZipFile(zpath) as z:
        txt_members = [m for m in z.namelist() if m.lower().endswith(".txt")]
        if not txt_members:
            dbg("OBS_ZIP_SKIP", f"no .txt in {os.path.basename(zpath)}")
            continue

        for member in txt_members:
            with z.open(member) as f:
                reader = pd.read_csv(
                    f,
                    sep="\t",
                    dtype=str,
                    usecols=usecols,
                    chunksize=chunksize
                )

                for chunk in reader:
                    total_chunks_read += 1
                    chunk_rows = len(chunk)
                    total_rows_read += chunk_rows

                    # normalize
                    chunk["medcodeid"] = chunk["medcodeid"].astype(str).str.strip()

                    # faster filter: small set first
                    chunk = chunk[chunk["medcodeid"].isin(ethnicity_set)]
                    if chunk.empty:
                        continue

                    eth_match_rows = len(chunk)
                    total_eth_matches += eth_match_rows

                    chunk["patid"] = chunk["patid"].astype(str)
                    chunk = chunk[chunk["patid"].isin(baseline_patids)]
                    if chunk.empty:
                        continue

                    baseline_match_rows = len(chunk)
                    total_baseline_matches += baseline_match_rows

                    # map
                    chunk["gen_ethnicity"] = chunk["medcodeid"].map(ethnicity_dict)

                    # date parse
                    chunk["obsdate"] = pd.to_datetime(chunk["obsdate"], errors="coerce", dayfirst=True)

                    # earliest per patid in this chunk WITHOUT global sort:
                    # groupby min obsdate per patid
                    min_dt = chunk.groupby("patid", sort=False)["obsdate"].min()

                    # pick ethnicity at that min date: sort small subset only
                    # (keep rows where obsdate == group min, then drop_duplicates)
                    chunk = chunk.merge(min_dt.rename("min_obsdate"), on="patid", how="left")
                    chunk = chunk[chunk["obsdate"] == chunk["min_obsdate"]]
                    chunk = chunk.drop_duplicates("patid", keep="first")[["patid", "obsdate", "gen_ethnicity"]]

                    for row in chunk.itertuples(index=False):
                        pat = row.patid
                        dt = row.obsdate
                        eth = row.gen_ethnicity
                        if pat not in best:
                            best[pat] = (dt, eth)
                        else:
                            if should_replace(best[pat][0], dt):
                                best[pat] = (dt, eth)

    if i % 10 == 0:
        elapsed = time.time() - loop_start
        rate = i / elapsed if elapsed > 0 else 0
        remaining = (len(obs_zips) - i) / rate if rate > 0 else float("inf")
        log(
            f"Processed {i}/{len(obs_zips)} obs ZIPs | "
            f"last_zip={time.time()-zip_start:.1f}s | "
            f"ETA~{remaining/3600:.2f}h | "
            f"best_size={len(best):,}"
        )
        dbg("OBS_PROGRESS", f"total_chunks={total_chunks_read:,}  total_rows={total_rows_read:,}  "
                            f"eth_matches={total_eth_matches:,}  baseline_matches={total_baseline_matches:,}")

log(f"Finished obs loop. best_size={len(best):,}")
dbg("OBS_LOOP_DONE", f"total_chunks={total_chunks_read:,}  total_rows_read={total_rows_read:,}")
dbg("OBS_LOOP_DONE", f"total_eth_matches={total_eth_matches:,}  total_baseline_matches={total_baseline_matches:,}")
dbg("OBS_LOOP_DONE", f"unique patients with CPRD ethnicity={len(best):,}")

# ethnicity group breakdown from best dict
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

# HES wins, CPRD fallback fills missing
eth_combined = (
    hes_eth.set_index("patid")
    .combine_first(obs_eth.set_index("patid"))
    .reset_index()
)
log(f"Ethnicity combined (HES priority) rows: n={eth_combined.shape[0]:,}")
dbg("ETH_COMBINED", f"gen_ethnicity value_counts:\n{eth_combined['gen_ethnicity'].value_counts(dropna=False).to_string()}")

# How many from HES vs CPRD vs still missing?
hes_patids = set(hes_eth["patid"].astype(str))
cprd_patids = set(obs_eth["patid"].astype(str))
eth_combined_patids = set(eth_combined["patid"].astype(str))
hes_only = hes_patids - cprd_patids
cprd_only = cprd_patids - hes_patids
both = hes_patids & cprd_patids
still_missing = baseline_patids - eth_combined_patids
has_eth = eth_combined[eth_combined["gen_ethnicity"].notna()]
still_na = baseline_patids - set(has_eth["patid"].astype(str))
dbg("ETH_SOURCE", f"HES_only={len(hes_only):,}  CPRD_only={len(cprd_only):,}  both={len(both):,}")
dbg("ETH_SOURCE", f"not_in_either_table={len(still_missing):,}  still_NA_ethnicity={len(still_na):,}")
dbg("ETH_SOURCE", f"total coverage: {len(baseline_patids)-len(still_na):,}/{len(baseline_patids):,} = "
                   f"{100*(len(baseline_patids)-len(still_na))/len(baseline_patids):.2f}%")


# -----------------------
# 4) IMD
# -----------------------
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
dbg("IMD_FILTERED", f"e2019_imd_10 value_counts:\n{imd['e2019_imd_10'].value_counts(dropna=False).sort_index().to_string()}")
dbg("IMD_FILTERED", f"coverage: {len(imd):,}/{len(baseline_patids):,} = "
                     f"{100*len(imd)/len(baseline_patids):.2f}%")


# -----------------------
# 5) Date of death (ONS only)
# -----------------------
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
dbg("DEATH_FILTERED", f"deaths with valid dod_ons={death_ons['dod_ons'].notna().sum():,}  "
                       f"missing dod_ons={death_ons['dod_ons'].isna().sum():,}")

death_ons["death_ons"] = ((death_ons["dod_ons"].notna()) & (death_ons["dod_ons"] <= STUDY_END)).astype(int)
death_ons["censor_date"] = death_ons["dod_ons"]
death_ons.loc[
    death_ons["censor_date"].isna() | (death_ons["censor_date"] > STUDY_END),
    "censor_date"
] = STUDY_END

dbg("DEATH_OUTCOME", f"death_ons=1 count: {(death_ons['death_ons']==1).sum():,}  "
                      f"death_ons=0 count: {(death_ons['death_ons']==0).sum():,}")
dbg("DEATH_OUTCOME", f"deaths AFTER study_end (censored): "
                      f"{(death_ons['dod_ons'].notna() & (death_ons['dod_ons'] > STUDY_END)).sum():,}")
dbg_date_col(death_ons, "censor_date", "DEATH_CENSOR")


# -----------------------
# 6) Merge demographics
# -----------------------
log("Merging demographics")
dbg("PRE_DEMO_MERGE", f"patient_all={len(patient_all):,}  eth_combined={len(eth_combined):,}  "
                       f"imd={len(imd):,}  death_ons={len(death_ons):,}")

demographics = (
    patient_all
    .merge(eth_combined, on="patid", how="left")
    .merge(imd, on="patid", how="left")
    .merge(death_ons[["patid", "dod_ons", "death_ons", "censor_date"]], on="patid", how="left")
)
log(f"Demographics merged: shape={demographics.shape}")
dbg_df(demographics, "DEMOGRAPHICS")
dbg("DEMOGRAPHICS", f"columns={list(demographics.columns)}")

# missingness report
for col in demographics.columns:
    n_miss = demographics[col].isna().sum()
    if n_miss > 0:
        dbg("DEMOGRAPHICS_MISSING", f"{col}: {n_miss:,} ({100*n_miss/len(demographics):.2f}%)")


# -----------------------
# 7) Enrich baseline and save TSV
# -----------------------
log("Merging demographics onto baseline")
pre_enrich_rows = len(baseline_df)
enriched = baseline_df.merge(demographics, on="patid", how="left")

dbg("ENRICH_MERGE", f"baseline_rows={pre_enrich_rows:,}  enriched_rows={len(enriched):,}  "
                     f"(should be equal for left join)")
if len(enriched) != pre_enrich_rows:
    dbg("ENRICH_MERGE_WARNING", f"ROW COUNT CHANGED! Possible duplicate patids in demographics. "
                                 f"diff={len(enriched)-pre_enrich_rows:,}")

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

# Final value counts for key categorical columns
for col in ["gender", "gen_ethnicity", "e2019_imd_10", "death_ons"]:
    if col in enriched.columns:
        dbg("ENRICHED_FINAL", f"{col} value_counts:\n{enriched[col].value_counts(dropna=False).to_string()}")

# Date diagnostics on final output
for col in enriched.columns:
    if "date" in col.lower() or col.lower() in ("obsdate", "indexdate", "dod_ons"):
        dbg_date_col(enriched, col, "ENRICHED_FINAL")

# Type distribution preserved?
if "type" in enriched.columns:
    dbg("ENRICHED_FINAL", f"type distribution:\n{enriched['type'].value_counts(dropna=False).to_string()}")

log(f"Writing output: {output_path}")
enriched.to_csv(output_path, sep="\t", index=False)
log("Saved successfully.")

dbg("OUTPUT", f"file={output_path}  size_bytes={os.path.getsize(output_path):,}")

log(f"TOTAL runtime: {(time.time() - t0)/3600:.2f} hours")
