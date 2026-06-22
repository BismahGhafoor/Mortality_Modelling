import pandas as pd
import os

# ──────────────────────────────────────────────────────────
# DEBUG HELPER
# ──────────────────────────────────────────────────────────
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
    if gained:
        sample = sorted(gained)[:print_max]
        print(f"DBG|   gained sample (≤{print_max}): {sample}")


# ──────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────
diabetes_code_doc = "/rfs/LRWE_Proj88/Shared/Codes/GOLD_final.txt"
out_dir = "/scratch/alice/b/bg205/28_02_GOLD"
out_path = os.path.join(out_dir, "filtered_diabetes_codes.txt")

# ──────────────────────────────────────────────────────────
# 1. LOAD  (read everything as str first for safety — matches AURUM)
# ──────────────────────────────────────────────────────────
df_codes = pd.read_csv(diabetes_code_doc, sep="\t", dtype=str, low_memory=False)

print("Loaded:", df_codes.shape)
print(f"DBG| [COLUMNS] {list(df_codes.columns)}")

# Identify the medcode column dynamically (matches AURUM)
code_col = None
for candidate in ["medcode", "MedCodeId", "medcodeid", "MedCode", "code"]:
    if candidate in df_codes.columns:
        code_col = candidate
        break
if code_col is None:
    raise KeyError(f"Cannot find a medcode column. Available: {list(df_codes.columns)}")
print(f"DBG| [CODE_COL] using '{code_col}' as the code column")

print("df_codes before mapping:")
print(df_codes["type"].value_counts(dropna=False))

dbg(df_codes, "LOAD_CODES")

# DBG: inspect raw type column before coercion
print(f"DBG| [TYPE_RAW] dtype={df_codes['type'].dtype}")
print(f"DBG| [TYPE_RAW] unique values (≤30): {sorted(df_codes['type'].dropna().unique())[:30]}")
print(f"DBG| [TYPE_RAW] NaN count={df_codes['type'].isna().sum():,}")

# ──────────────────────────────────────────────────────────
# 2. COERCE type TO NUMERIC  ← filter point: non-numeric → NaN
# ──────────────────────────────────────────────────────────
before_codes = set(df_codes[code_col].dropna().unique())

df_codes["type"] = pd.to_numeric(df_codes["type"], errors="coerce")

# DBG: how many NaN AFTER coercion?
nan_after_coerce = df_codes["type"].isna().sum()
print(f"DBG| [TYPE_COERCE] NaN after pd.to_numeric coerce={nan_after_coerce:,}")
if nan_after_coerce > 0:
    print(f"DBG| [TYPE_COERCE] ⚠ {nan_after_coerce:,} codes will be LOST at the isin filter!")

# ──────────────────────────────────────────────────────────
# 3. REMAP 0 → 2
# ──────────────────────────────────────────────────────────
df_codes["type"] = df_codes["type"].replace(0, 2)

print("\ndf_codes after mapping Type 0 -> Type 2:")
print(df_codes["type"].value_counts(dropna=False))

# ──────────────────────────────────────────────────────────
# 4. FILTER: keep type 1 & 2  ← main filter point
# ──────────────────────────────────────────────────────────
dbg(df_codes, "PRE_ISIN_FILTER")

df_filtered_codes = df_codes.loc[df_codes["type"].isin([1, 2])].copy()

dbg(df_filtered_codes, "POST_ISIN_FILTER")

after_codes = set(df_filtered_codes[code_col].dropna().unique())
dbg_set_diff(before_codes, after_codes, "ISIN_FILTER_codes")

# DBG: what types were dropped?
dropped = df_codes.loc[~df_codes["type"].isin([1, 2])]
print(f"DBG| [ISIN_DROPPED] rows dropped={len(dropped):,}")
print(f"DBG| [ISIN_DROPPED] type distribution of dropped rows:")
print(f"DBG|   {dropped['type'].value_counts(dropna=False).to_dict()}")

print("\ndf_codes after filtering:")
print(df_filtered_codes["type"].value_counts(dropna=False))
print("Filtered shape:", df_filtered_codes.shape)

# ──────────────────────────────────────────────────────────
# 5. ADD TERMINOLOGY + RENAME
# ──────────────────────────────────────────────────────────
df_filtered_codes["terminology"] = "medcode"
df_filtered_codes.rename(columns={code_col: "code"}, inplace=True)

# Strip whitespace AFTER rename (matches AURUM timing)
df_filtered_codes["code"] = df_filtered_codes["code"].astype(str).str.strip()

# DBG: final code-list summary
dbg(df_filtered_codes, "FINAL_CODELIST")
print(f"DBG| [FINAL_CODELIST] unique codes={df_filtered_codes['code'].nunique():,}")
print(f"DBG| [FINAL_CODELIST] type split: {df_filtered_codes['type'].value_counts().to_dict()}")

# ──────────────────────────────────────────────────────────
# 6. SAVE
# ──────────────────────────────────────────────────────────
os.makedirs(out_dir, exist_ok=True)
df_filtered_codes.to_csv(out_path, sep="\t", index=False)

print(f"\nSaved to: {out_path}")
print(df_filtered_codes.head())
