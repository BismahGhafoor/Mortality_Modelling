#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SCRIPT 8 — LAB EXTRACTION WITH 1-YEAR WINDOW + CORRECT UNIT-CODE HANDLING + TC (PANEL-FIRST) + RECORDED FALLBACK
"""

import os
import pandas as pd
import numpy as np
import tempfile

print("Starting lab data extraction (TSV)…")

from helper_functions import lcf, ucf, perc
from helper_functions import save_long_format_data, read_long_format_data
from helper_functions import remap_eth, nperc_counts, calc_gfr

save_long_format = False
#WINDOW_DAYS = 1095
WINDOW_DAYS = 365
temp_dir = tempfile.TemporaryDirectory()
print("Temporary directory:", temp_dir.name)


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


def dbg_patid_diff(before_patids, after_patids, step_name):
    lost = before_patids - after_patids
    print(f"DBG|[{step_name}] patids before={len(before_patids):,}  after={len(after_patids):,}  "
          f"lost={len(lost):,}")
    if lost and len(lost) <= 20:
        print(f"DBG|  lost (first 20): {sorted(lost)[:20]}")
    elif lost:
        print(f"DBG|  lost (first 20 of {len(lost):,}): {sorted(lost)[:20]}")


def dbg_value_stats(series, name):
    """Quick numeric stats for a measurement column."""
    num = pd.to_numeric(series, errors='coerce')
    valid = num.dropna()
    if len(valid) == 0:
        print(f"DBG|[{name}] all NaN")
        return
    print(f"DBG|[{name}] n={len(valid):,}  min={valid.min():.3f}  p25={valid.quantile(.25):.3f}  "
          f"p50={valid.median():.3f}  p75={valid.quantile(.75):.3f}  max={valid.max():.3f}")


# ----------------------------------------------------------------------
# Load unit lookup
# ----------------------------------------------------------------------
UNITS_PATH = "/scratch/alice/b/bg205/DataCleaning_Gold_v2/SUM.txt"
units_df = pd.read_csv(UNITS_PATH, sep="\t", header=0)
units_df["Code"] = pd.to_numeric(units_df["Code"], errors="coerce").astype("Int64")

unit_name_col = "Specimen Unit of Measure" if "Specimen Unit of Measure" in units_df.columns else units_df.columns[-1]
_name = units_df[unit_name_col].astype(str).str.lower()
lipid_mmol_codes = set(
    units_df.loc[_name.str.contains("mmol") & _name.str.contains("/l"), "Code"]
    .dropna().astype(int).tolist()
)
print(f"DBG|[unit_lookup] lipid_mmol_codes={lipid_mmol_codes}")

lipid_mgdl_codes = set(
    units_df.loc[_name.str.contains("mg") & (_name.str.contains("/dl") | _name.str.contains("per dl")), "Code"]
    .dropna().astype(int).tolist()
)
print(f"DBG|[unit_lookup] lipid_mgdl_codes={lipid_mgdl_codes}")

hba1c_pct_codes     = {1, 215}
hba1c_mmolmol_codes = {97, 156, 205, 187}
hba1c_dcct96_codes  = {96}
print(f"DBG|[unit_lookup] hba1c: pct={hba1c_pct_codes}  mmol/mol={hba1c_mmolmol_codes}  dcct96={hba1c_dcct96_codes}")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _pick_closest_within_window_before_or_at(df, date_col="eventdate"):
    if df is None or df.empty:
        return df

    x = df.copy()
    pats_input = set(x['patid'].unique())

    x = x[x[date_col].notna() & x["indexdate"].notna()]
    pats_after_notna = set(x['patid'].unique())

    x = x[x[date_col] <= x["indexdate"]]
    pats_after_leq = set(x['patid'].unique())

    if x.empty:
        print(f"DBG|[_pick_window] empty after eventdate<=indexdate")
        return x

    x["gap_days"] = (x["indexdate"] - x[date_col]).dt.days
    pats_before_window = set(x['patid'].unique())
    x = x[(x["gap_days"] >= 0) & (x["gap_days"] <= WINDOW_DAYS)]
    pats_after_window = set(x['patid'].unique())

    print(f"DBG|[_pick_window] patids: input={len(pats_input):,} -> notna={len(pats_after_notna):,} "
          f"-> <=idx={len(pats_after_leq):,} -> {WINDOW_DAYS}d_window={len(pats_after_window):,}  "
          f"lost_by_window={len(pats_before_window - pats_after_window):,}")

    if x.empty:
        return x

    x = x.sort_values(["patid", "indexdate", "gap_days", date_col],
                      ascending=[True, True, True, False])
    x = x.drop_duplicates(subset=["patid", "indexdate"], keep="first").reset_index(drop=True)

    print(f"DBG|[_pick_window] final best: {len(x):,} patids  "
          f"gap_days: min={x['gap_days'].min()}  p50={x['gap_days'].median():.0f}  max={x['gap_days'].max()}")

    return x


def _detect_measurement_and_unit_code(df_sub: pd.DataFrame):
    d1 = pd.to_numeric(df_sub.get("data1"), errors="coerce")
    d2 = pd.to_numeric(df_sub.get("data2"), errors="coerce")
    d3 = pd.to_numeric(df_sub.get("data3"), errors="coerce")

    frac_d1 = (d1 % 1 != 0).mean(skipna=True) if d1 is not None else 0
    frac_d2 = (d2 % 1 != 0).mean(skipna=True) if d2 is not None else 0
    frac_d3 = (d3 % 1 != 0).mean(skipna=True) if d3 is not None else 0

    def code_likeness(x):
        if x is None:
            return 0.0
        xi = x.dropna()
        if xi.empty:
            return 0.0
        is_int = (xi % 1 == 0)
        small = (xi >= 0) & (xi <= 500)
        return float((is_int & small).mean())

    cl_d1 = code_likeness(d1)
    cl_d2 = code_likeness(d2)
    cl_d3 = code_likeness(d3)

    # Prefer the most plausible quantitative layout:
    # data2 = measurement, data3 = unit code
    if frac_d2 >= max(frac_d1, frac_d3) and cl_d3 >= max(cl_d1, cl_d2):
        meas = d2
        code = d3.round(0).astype("Int64")
        src = "data2->measurement, data3->unit_code"
    # fallback: data3 looks more like measurement, data2 looks more like code
    elif frac_d3 > frac_d2 and cl_d2 >= max(cl_d1, cl_d3):
        meas = d3
        code = d2.round(0).astype("Int64")
        src = "data3->measurement, data2->unit_code"
    # fallback: use data2 as measurement and no reliable code
    else:
        meas = d2
        code = d3.round(0).astype("Int64")
        src = "fallback data2->measurement, data3->unit_code"

    print(
        f"[detect] {src}; "
        f"frac(d1/d2/d3)={frac_d1:.2f}/{frac_d2:.2f}/{frac_d3:.2f}; "
        f"code_like(d1/d2/d3)={cl_d1:.2f}/{cl_d2:.2f}/{cl_d3:.2f}"
    )

    dbg_value_stats(meas, "detected_measurement")
    code_vc = pd.to_numeric(code, errors="coerce").dropna().astype(int).value_counts().head(10)
    print(f"DBG|[detected_unit_code] top codes:\n{code_vc}")

    return meas, code


def _merge_selected(patient, df_sel, value_col, date_col_name):
    if df_sel is None or df_sel.empty:
        patient[date_col_name] = pd.NaT
        patient[value_col] = np.nan
        return patient

    z = df_sel.rename(columns={"eventdate": date_col_name})
    before_rows = len(patient)
    out = patient.merge(z[["patid", "indexdate", date_col_name, value_col]],
                        on=["patid", "indexdate"], how="left")

    if len(out) != before_rows:
        print(f"DBG|[_merge_selected_{value_col}] WARNING: row change {before_rows:,} -> {len(out):,}")

    return out


def _unit_ok(series_int):
    return (
        series_int.isin(lipid_mmol_codes) |
        (series_int == 96) |
        (series_int.isna()) |
        (series_int == 0)
    )

MIN_DATE = pd.Timestamp("1900-01-01")
MAX_DATE = pd.Timestamp("2024-12-31")

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
# Generic lipid extractor
# ----------------------------------------------------------------------
def extract_lipid(test, patient, enttype, outcol, min_val, max_val):
    enttype = str(enttype)
    print(f"\n{'─'*60}")
    print(f"[extract_lipid] {outcol} enttype={enttype} range=[{min_val},{max_val}] mmol/L window={WINDOW_DAYS}d")
    print(f"{'─'*60}")

    df = test[test["enttype"] == enttype].copy()
    if df.empty:
        print("[extract_lipid] No rows.")
        patient[outcol] = np.nan
        patient[f"{outcol}_date"] = pd.NaT
        return patient

    dbg(df, f"{outcol}_raw_enttype_{enttype}")

    # Map indexdate from patient
    df["indexdate"] = df["patid"].map(patient.set_index("patid")["indexdate"])

    df["eventdate"] = pd.to_datetime(df["eventdate"], errors="coerce", format="%Y-%m-%d")
    df = enforce_date_bounds(df, "eventdate")
    df["indexdate"] = pd.to_datetime(df["indexdate"], errors="coerce", format="%Y-%m-%d")

    # ── DBG: date parsing check ──
    n_eventdate_nat = df["eventdate"].isna().sum()
    n_indexdate_nat = df["indexdate"].isna().sum()
    print(f"DBG|[{outcol}_dates] eventdate NaT={n_eventdate_nat:,}  indexdate NaT={n_indexdate_nat:,}  "
          f"(indexdate NaT = non-baseline patients, expected)")

    meas, code = _detect_measurement_and_unit_code(df)
    df[outcol] = pd.to_numeric(meas, errors="coerce")
    df["unit_code_int"] = pd.to_numeric(code, errors="coerce").astype("Int64")

    # ── DBG: step-by-step filter tracking ──
    pats_start = set(df.loc[df[outcol].notna(), 'patid'].unique())
    before = len(df)

    df = df[df["eventdate"].notna() & df["indexdate"].notna() & df[outcol].notna()]
    after_notna = len(df)
    pats_after_notna = set(df['patid'].unique())
    print(f"DBG|[{outcol}_filter] after dropna(date+value): {before:,} -> {after_notna:,}  "
          f"patids: {len(pats_start):,} -> {len(pats_after_notna):,}")

    # Convert mg/dL to mmol/L (matches AURUM)
    is_mmol = df["unit_code_int"].isin(lipid_mmol_codes)
    is_mgdl = df["unit_code_int"].isin(lipid_mgdl_codes)
    is_na_or_zero = df["unit_code_int"].isna() | (df["unit_code_int"] == 0)
    is_unknown = df["unit_code_int"].notna() & ~is_mmol & ~is_mgdl & ~is_na_or_zero

    if outcol in ("trigly",):
        factor = 0.01129
    else:
        factor = 0.02586

    n_converted = int(is_mgdl.sum())
    if n_converted > 0:
        df.loc[is_mgdl, outcol] = df.loc[is_mgdl, outcol] * factor

    print(f"DBG|[{outcol}_unit_conv] mmol/L={is_mmol.sum():,}  mg/dL_converted={n_converted:,}  "
          f"no_unit/zero={is_na_or_zero.sum():,}  unknown_dropped={is_unknown.sum():,}")

    before_unit = len(df)
    pats_before_unit = set(df['patid'].unique())
    df = df[~is_unknown]
    pats_after_unit = set(df['patid'].unique())
    print(f"DBG|[{outcol}_filter] after dropping unknown units: {before_unit:,} -> {len(df):,}  "
          f"patids lost={len(pats_before_unit - pats_after_unit):,}")
    
    before_range = len(df)
    pats_before_range = set(df['patid'].unique())
    df = df[(df[outcol] >= float(min_val)) & (df[outcol] <= float(max_val))]
    pats_after_range = set(df['patid'].unique())
    print(f"DBG|[{outcol}_filter] after range [{min_val},{max_val}]: {before_range:,} -> {len(df):,}  "
          f"patids lost={len(pats_before_range - pats_after_range):,}")
    if pats_before_range - pats_after_range:
        # show what values those lost patients had
        lost_pats = pats_before_range - pats_after_range
        if len(lost_pats) <= 5:
            print(f"DBG|  lost patids: {sorted(lost_pats)}")

    print(f"[extract_lipid] After plausibility+unit filter: kept {len(df):,} / {before:,}")
    if df.empty:
        patient[outcol] = np.nan
        patient[f"{outcol}_date"] = pd.NaT
        return patient

    df = df.groupby(["patid", "indexdate", "eventdate"], as_index=False)[outcol].mean()

    print(f"\n--- {outcol}: pick closest within {WINDOW_DAYS}d window ---")
    df_sel = _pick_closest_within_window_before_or_at(df[["patid", "indexdate", "eventdate", outcol]].copy())

    if save_long_format:
        save_long_format_data(df[["patid", "eventdate", "indexdate", outcol]].copy(), True, outcol)

    # ── DBG: value distribution of selected measurements ──
    if df_sel is not None and not df_sel.empty:
        dbg_value_stats(df_sel[outcol], f"{outcol}_selected")

    patient = _merge_selected(patient, df_sel, value_col=outcol, date_col_name=f"{outcol}_date")
    have = int(pd.to_numeric(patient[outcol], errors="coerce").notna().sum())
    miss = len(patient) - have
    print(f"[extract_lipid] Merged {outcol}: {have:,} / {len(patient):,}  missing={miss:,} ({miss/len(patient)*100:.2f}%)")
    return patient


# ----------------------------------------------------------------------
# Total cholesterol: PANEL-FIRST
# ----------------------------------------------------------------------
def extract_total_cholesterol_panel_first(test, patient):
    print(f"\n{'='*60}")
    print("[TC] Building TC from same-day lipid panels (preferred)…")
    print(f"{'='*60}")

    def _prep_ent(enttype, colname, min_val, max_val):
        d = test[test["enttype"] == str(enttype)].copy()
        if d.empty:
            return d

        d["indexdate"] = d["patid"].map(patient.set_index("patid")["indexdate"])
        d["eventdate"] = pd.to_datetime(d["eventdate"], errors="coerce", format="%Y-%m-%d")
        d = enforce_date_bounds(d, "eventdate")
        d["indexdate"] = pd.to_datetime(d["indexdate"], errors="coerce", format="%Y-%m-%d")

        meas, code = _detect_measurement_and_unit_code(d)
        d[colname] = pd.to_numeric(meas, errors="coerce")
        d["unit_code_int"] = pd.to_numeric(code, errors="coerce").astype("Int64")

        before = len(d)
        d = d[d["eventdate"].notna() & d["indexdate"].notna() & d[colname].notna()]
        is_mmol = d["unit_code_int"].isin(lipid_mmol_codes)
        is_mgdl = d["unit_code_int"].isin(lipid_mgdl_codes)
        is_na_or_zero = d["unit_code_int"].isna() | (d["unit_code_int"] == 0)
        is_unknown = d["unit_code_int"].notna() & ~is_mmol & ~is_mgdl & ~is_na_or_zero

        if colname in ("tg_tmp",):
            factor = 0.01129
        else:
            factor = 0.02586

        if is_mgdl.sum() > 0:
            d.loc[is_mgdl, colname] = d.loc[is_mgdl, colname] * factor
            print(f"DBG|[TC_prep_{colname}] converted {is_mgdl.sum():,} mg/dL rows")

        d = d[~is_unknown]
        
        d = d[(d[colname] >= float(min_val)) & (d[colname] <= float(max_val))]

        d = d[d["eventdate"] <= d["indexdate"]].copy()
        d["gap_days"] = (d["indexdate"] - d["eventdate"]).dt.days
        d = d[(d["gap_days"] >= 0) & (d["gap_days"] <= WINDOW_DAYS)].copy()

        print(f"DBG|[TC_prep_{colname}] enttype={enttype}: {before:,} -> {len(d):,}  "
              f"patids={d['patid'].nunique() if len(d) > 0 else 0:,}")

        if d.empty:
            return d

        d = d.groupby(["patid", "indexdate", "eventdate"], as_index=False)[colname].mean()
        return d

    ldl = _prep_ent("177", "ldl_tmp", 0, 20)
    hdl = _prep_ent("175", "hdl_tmp", 0, 10)
    tg  = _prep_ent("202", "tg_tmp",  0, 40)

    if ldl.empty or hdl.empty or tg.empty:
        triple_best = pd.DataFrame(columns=["patid", "indexdate", "eventdate", "tot_chol_calc"])
        print("[TC] One of LDL/HDL/TG is empty in-window → no TC_calc panels.")
    else:
        # ── DBG: before inner joins ──
        ldl_pats = set(ldl['patid'].unique())
        hdl_pats = set(hdl['patid'].unique())
        tg_pats  = set(tg['patid'].unique())
        all_three = ldl_pats & hdl_pats & tg_pats
        print(f"DBG|[TC_panel] patients with in-window data: LDL={len(ldl_pats):,}  HDL={len(hdl_pats):,}  "
              f"TG={len(tg_pats):,}  all_three={len(all_three):,}")

        triple = ldl.merge(hdl, on=["patid", "indexdate", "eventdate"], how="inner")
        triple = triple.merge(tg,  on=["patid", "indexdate", "eventdate"], how="inner")

        print(f"DBG|[TC_panel] same-day triples: {len(triple):,} rows  "
              f"patids={triple['patid'].nunique() if len(triple) > 0 else 0:,}")

        if triple.empty:
            triple_best = pd.DataFrame(columns=["patid", "indexdate", "eventdate", "tot_chol_calc"])
            print("[TC] No same-day LDL+HDL+TG triples found in-window.")
        else:
            triple["tot_chol_calc"] = triple["ldl_tmp"] + triple["hdl_tmp"] + (triple["tg_tmp"] / 2.2)
            dbg_value_stats(triple["tot_chol_calc"], "TC_calc_pre_plausibility")

            before_plaus = len(triple)
            triple = triple[(triple["tot_chol_calc"] >= 0) & (triple["tot_chol_calc"] <= 20)].copy()
            print(f"DBG|[TC_panel] after plausibility [0,20]: {before_plaus:,} -> {len(triple):,}")

            if triple.empty:
                triple_best = pd.DataFrame(columns=["patid", "indexdate", "eventdate", "tot_chol_calc"])
                print("[TC] TC_calc plausibility removed all triples.")
            else:
                triple["gap_days"] = (triple["indexdate"] - triple["eventdate"]).dt.days
                triple = triple.sort_values(["patid", "indexdate", "gap_days", "eventdate"],
                                            ascending=[True, True, True, False])
                triple_best = triple.drop_duplicates(subset=["patid", "indexdate"], keep="first").copy()
                print(f"[TC] TC_calc available for {len(triple_best):,} patient-indexdates.")
                dbg_value_stats(triple_best["tot_chol_calc"], "TC_calc_selected")

    # Merge TC_calc first
    out = patient.copy()
    if not triple_best.empty:
        triple_best = triple_best.rename(columns={"eventdate": "tot_chol_calc_date"})
        out = out.merge(triple_best[["patid", "indexdate", "tot_chol_calc", "tot_chol_calc_date"]],
                        on=["patid", "indexdate"], how="left")
    else:
        out["tot_chol_calc"] = np.nan
        out["tot_chol_calc_date"] = pd.NaT

    # Extract recorded TC (fallback)
    print(f"\n{'─'*60}")
    print("[TC] Extracting recorded total cholesterol (fallback)…")
    print(f"{'─'*60}")
    out = extract_lipid(test, out, enttype="163", outcol="tot_chol_rec", min_val=0, max_val=20)

    # PANEL-FIRST selection
    out["tot_chol"] = np.nan
    out["tot_chol_date"] = pd.NaT
    out["tot_chol_source"] = pd.NA

    use_calc = pd.to_numeric(out["tot_chol_calc"], errors="coerce").notna()
    out.loc[use_calc, "tot_chol"] = out.loc[use_calc, "tot_chol_calc"]
    out.loc[use_calc, "tot_chol_date"] = out.loc[use_calc, "tot_chol_calc_date"]
    out.loc[use_calc, "tot_chol_source"] = "calculated_LDL+HDL+TG/2.2"

    use_rec = (~use_calc) & pd.to_numeric(out["tot_chol_rec"], errors="coerce").notna()
    out.loc[use_rec, "tot_chol"] = out.loc[use_rec, "tot_chol_rec"]
    out.loc[use_rec, "tot_chol_date"] = out.loc[use_rec, "tot_chol_rec_date"]
    out.loc[use_rec, "tot_chol_source"] = "recorded"

    out = out.drop(columns=["tot_chol_calc", "tot_chol_calc_date", "tot_chol_rec", "tot_chol_rec_date"],
                   errors="ignore")

    have = int(pd.to_numeric(out["tot_chol"], errors="coerce").notna().sum())
    miss = len(out) - have
    print(f"\n[TC final] tot_chol available: {have:,} / {len(out):,}  missing={miss:,} ({miss/len(out)*100:.2f}%)")
    print("[TC final] source counts:")
    print(out["tot_chol_source"].value_counts(dropna=False).head(10).to_string())
    return out


# ----------------------------------------------------------------------
# HbA1c extractor
# ----------------------------------------------------------------------
def extract_hba1c(test, patient):
    print(f"\n{'='*60}")
    print("[extract_hba1c] enttype=275 windowed before/at indexdate")
    print(f"{'='*60}")

    h = test[test["enttype"] == "275"].copy()
    if h.empty:
        patient["hba1c_perc"] = np.nan
        patient["hba1c_date"] = pd.NaT
        return patient

    dbg(h, "hba1c_raw_enttype_275")

    h["indexdate"] = h["patid"].map(patient.set_index("patid")["indexdate"])
    h["eventdate"] = pd.to_datetime(h["eventdate"], errors="coerce", format="%Y-%m-%d")
    h = enforce_date_bounds(h, "eventdate")
    h["indexdate"] = pd.to_datetime(h["indexdate"], errors="coerce", format="%Y-%m-%d")

    n_eventdate_nat = h["eventdate"].isna().sum()
    n_indexdate_nat = h["indexdate"].isna().sum()
    print(f"DBG|[hba1c_dates] eventdate NaT={n_eventdate_nat:,}  indexdate NaT={n_indexdate_nat:,}")

    meas, code = _detect_measurement_and_unit_code(h)
    h["value_num"] = pd.to_numeric(meas, errors="coerce")
    h["unit_int"]  = pd.to_numeric(code, errors="coerce").round(0).astype("Int64")

    # ── DBG: unit code distribution for HbA1c ──
    unit_vc = h["unit_int"].value_counts(dropna=False).head(15)
    print(f"DBG|[hba1c_unit_codes] distribution:\n{unit_vc}")

    recognised_codes = hba1c_pct_codes | hba1c_mmolmol_codes | hba1c_dcct96_codes
    n_unrecognised = (~h["unit_int"].isin(recognised_codes) & h["unit_int"].notna()).sum()
    n_unit_na = h["unit_int"].isna().sum()
    print(f"DBG|[hba1c_unit_codes] recognised={h['unit_int'].isin(recognised_codes).sum():,}  "
          f"unrecognised={n_unrecognised:,}  unit_NA={n_unit_na:,}")
    if n_unrecognised > 0:
        unrec = h.loc[~h["unit_int"].isin(recognised_codes) & h["unit_int"].notna(), "unit_int"]
        print(f"DBG|  unrecognised codes: {unrec.value_counts().head(10).to_dict()}")

    # Convert to %
    h["hba1c_perc"] = np.nan
    h.loc[h["unit_int"].isin(hba1c_pct_codes),     "hba1c_perc"] = h["value_num"]
    h.loc[h["unit_int"].isin(hba1c_mmolmol_codes), "hba1c_perc"] = h["value_num"] * 0.0915 + 2.15
    h.loc[h["unit_int"].isin(hba1c_dcct96_codes),  "hba1c_perc"] = h["value_num"] * 0.6277 + 1.627

    n_converted = h["hba1c_perc"].notna().sum()
    n_no_convert = h["hba1c_perc"].isna().sum()
    print(f"DBG|[hba1c_conversion] converted={n_converted:,}  not_converted(unit not recognised)={n_no_convert:,}")
    if n_converted > 0:
        dbg_value_stats(h["hba1c_perc"], "hba1c_perc_pre_plausibility")

    pats_before_plaus = set(h.loc[h["hba1c_perc"].notna(), 'patid'].unique())
    h = h[h["eventdate"].notna() & h["indexdate"].notna() & h["hba1c_perc"].notna()].copy()

    before = len(h)
    h = h[(h["hba1c_perc"] > 2.0) & (h["hba1c_perc"] <= 20.0)].copy()
    pats_after_plaus = set(h['patid'].unique())
    print(f"[extract_hba1c] After plausibility: kept {len(h):,} / {before:,}  "
          f"patids lost by range={len(pats_before_plaus - pats_after_plaus):,}")

    if h.empty:
        patient["hba1c_perc"] = np.nan
        patient["hba1c_date"] = pd.NaT
        return patient

    h_small = h[["patid", "indexdate", "eventdate", "hba1c_perc"]].copy()

    print(f"\n--- HbA1c: pick closest within {WINDOW_DAYS}d window ---")
    h_sel = _pick_closest_within_window_before_or_at(h_small, date_col="eventdate")

    if save_long_format:
        save_long_format_data(h[["patid", "eventdate", "indexdate", "hba1c_perc"]].copy(), True, "hba1c")

    if h_sel is not None and not h_sel.empty:
        dbg_value_stats(h_sel["hba1c_perc"], "hba1c_perc_selected")

    patient = _merge_selected(patient, h_sel, value_col="hba1c_perc", date_col_name="hba1c_date")
    have = int(pd.to_numeric(patient["hba1c_perc"], errors="coerce").notna().sum())
    miss = len(patient) - have
    print(f"[extract_hba1c] Merged: {have:,} / {len(patient):,}  missing={miss:,} ({miss/len(patient)*100:.2f}%)")
    return patient


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n[Main] Reading patient TSV…")
    patient = pd.read_csv(
        "/scratch/alice/b/bg205/28_02_GOLD/Cleaned_Patient_Smoking_BMI_BP_Data_3YEAR.txt",
        sep="\t", dtype=str
    )
    patient["patid"] = patient["patid"].astype(str)

    for c in ["indexdate", "smoking_date", "bmi_date", "bp_date", "dod"]:
        if c in patient.columns:
            patient[c] = pd.to_datetime(patient[c], errors="coerce", format="%Y-%m-%d")

    dbg(patient, "patient_baseline_loaded", date_cols=["indexdate"])
    checkpoint_baseline = set(patient["patid"].unique())
    print(f"[Main] Patient rows: {len(patient):,}")

    print("\n[Main] Reading Test_entities_all.txt.gz in chunks…")
    usecols = ["patid", "enttype", "data1", "data2", "data3", "eventdate"]

    chunks = []
    total_in = 0
    total_kept = 0
    for i, ch in enumerate(pd.read_csv(
        "Test_entities_all.txt.gz",
        sep="\t", compression="gzip", header=0,
        usecols=usecols,
        dtype=str,
        chunksize=200_000
    ), start=1):
        ch["patid"] = ch["patid"].astype(str)
        filt = ch[ch["patid"].isin(checkpoint_baseline)].copy()
        total_in += len(ch)
        total_kept += len(filt)
        chunks.append(filt)
        if i % 100 == 0:
            print(f"  chunk {i}: cumulative in={total_in:,} kept={total_kept:,}")

    test = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    print(f"\n[Main] Test data: total_read={total_in:,}  kept(baseline)={total_kept:,}  "
          f"dropped(non-baseline)={total_in - total_kept:,}")
    dbg(test, "test_filtered_to_baseline")

    # ── DBG: enttype distribution in filtered test data ──
    print(f"DBG|[test_enttypes] distribution:\n{test['enttype'].value_counts().head(20)}")

    if test.empty:
        raise RuntimeError("Test file read produced an empty dataframe after filtering to patient IDs.")

    test["enttype"] = test["enttype"].astype(str)

    # Extract lipids
    patient = extract_lipid(test, patient, enttype="177", outcol="ldl",    min_val=0, max_val=20)
    patient = extract_lipid(test, patient, enttype="175", outcol="hdl",    min_val=0, max_val=10)
    patient = extract_lipid(test, patient, enttype="202", outcol="trigly", min_val=0, max_val=40)

    # Total cholesterol (panel-first; recorded fallback)
    patient = extract_total_cholesterol_panel_first(test, patient)

    # HbA1c
    patient = extract_hba1c(test, patient)

    # ── DBG: final missingness summary ──
    print(f"\n{'='*60}")
    print("FINAL LAB MISSINGNESS SUMMARY")
    print(f"{'='*60}")
    final_pats = set(patient['patid'].unique())
    dbg_patid_diff(checkpoint_baseline, final_pats, "baseline_vs_final_lab")

    for col in ["ldl", "hdl", "trigly", "tot_chol", "hba1c_perc"]:
        if col in patient.columns:
            n_miss = patient[col].isna().sum()
            pct = n_miss / len(patient) * 100
            have = len(patient) - n_miss
            print(f"DBG|[final_lab_missing] {col}: have={have:,}  missing={n_miss:,} ({pct:.2f}%)")

    # ── DBG: tot_chol source breakdown ──
    if "tot_chol_source" in patient.columns:
        print(f"DBG|[tot_chol_source] breakdown:\n{patient['tot_chol_source'].value_counts(dropna=False)}")

    # Save
    final_path = "/scratch/alice/b/bg205/28_02_GOLD/extracted_lab_data_1YEAR.txt"
    patient.to_csv(final_path, sep="\t", index=False, date_format="%Y-%m-%d")
    print(f"\n[Main] Saved: {final_path}")
    print(f"DBG|[SCRIPT8_FINAL] rows={len(patient):,}  patids={patient['patid'].nunique():,}")
