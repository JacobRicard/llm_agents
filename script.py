"""
export_brand_csvs.py
--------------------
Reads Structured_Data.db and exports one CSV per major car brand.
Each CSV is produced via a JOIN between the makes and models tables,
plus UNION queries that combine data across years and brands for
richer structured data (good for the SQL agent).
 
Usage:
    python export_brand_csvs.py
 
Edit the paths below if needed.
"""
 
import os
import sqlite3
import pandas as pd
 
# ── Paths ──────────────────────────────────────────────────────────────────────
DB_PATH    = r"C:\Users\Jacob\OneDrive\Junior Year Second Semester\AIM 4420\Structured_Data.db"
OUTPUT_DIR = r"C:\Users\Jacob\OneDrive\Junior Year Second Semester\AIM 4420\llm_agents\data"
 
# Major consumer car brands to export
BRANDS = [
    "BMW",
    "AUDI",
    "MERCEDES-BENZ",
    "VOLKSWAGEN",
    "PORSCHE",
    "TOYOTA",
    "HONDA",
    "NISSAN",
    "MAZDA",
    "SUBARU",
    "MITSUBISHI",
    "LEXUS",
    "CHEVROLET",
    "FORD",
    "DODGE",
    "JEEP",
    "KIA",
    "HYUNDAI",
]
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
conn = sqlite3.connect(DB_PATH)
 
# ── 1. One CSV per brand (JOIN makes + models, raw one row per model per year) ──
print("Exporting per-brand CSVs...")
for brand in BRANDS:
    query = """
        SELECT
            mo.make_id,
            ma.make_name,
            mo.model_id,
            mo.model_name,
            mo.year
        FROM models mo
        JOIN makes ma ON mo.make_id = ma.make_id
        WHERE mo.make_name = ?
        ORDER BY mo.year, mo.model_name
    """
    df = pd.read_sql_query(query, conn, params=(brand,))
 
    if df.empty:
        print(f"  Skipping {brand} — no data found")
        continue
 
    filename = brand.replace(" ", "_").replace("-", "_") + ".csv"
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
    print(f"  {brand}: {len(df)} rows → {filename}")
 
# ── 2. All brands combined CSV (one row per model per year) ────────────────────
print("\nExporting all_brands_combined.csv...")
placeholders = ",".join("?" * len(BRANDS))
union_query = f"""
    SELECT
        mo.make_id,
        ma.make_name,
        mo.model_id,
        mo.model_name,
        mo.year
    FROM models mo
    JOIN makes ma ON mo.make_id = ma.make_id
    WHERE mo.make_name IN ({placeholders})
    ORDER BY ma.make_name, mo.year, mo.model_name
"""
df_all = pd.read_sql_query(union_query, conn, params=BRANDS)
df_all.to_csv(os.path.join(OUTPUT_DIR, "all_brands_combined.csv"), index=False)
print(f"  all_brands_combined.csv: {len(df_all)} rows")
 
# ── 3. Model count summary per brand per decade ──────────────────────────────────
print("\nExporting brand_decade_summary.csv...")
summary_query = f"""
    SELECT
        mo.make_name,
        (mo.year / 10) * 10 AS decade,
        COUNT(DISTINCT mo.model_name) AS unique_models,
        COUNT(*) AS total_entries,
        MIN(mo.year) AS first_year,
        MAX(mo.year) AS last_year
    FROM models mo
    JOIN makes ma ON mo.make_id = ma.make_id
    WHERE mo.make_name IN ({placeholders})
    GROUP BY mo.make_name, decade
    ORDER BY mo.make_name, decade
"""
df_summary = pd.read_sql_query(summary_query, conn, params=BRANDS)
df_summary.to_csv(os.path.join(OUTPUT_DIR, "brand_decade_summary.csv"), index=False)
print(f"  brand_decade_summary.csv: {len(df_summary)} rows")
 
# ── 4. UNION example: German vs Japanese brands comparison ─────────────────────
print("\nExporting german_vs_japanese.csv...")
german   = ["BMW", "AUDI", "MERCEDES-BENZ", "VOLKSWAGEN", "PORSCHE"]
japanese = ["TOYOTA", "HONDA", "NISSAN", "MAZDA", "SUBARU", "MITSUBISHI", "LEXUS"]
 
gvj_query = """
    SELECT
        mo.make_name,
        'German' AS origin,
        mo.model_name,
        mo.year
    FROM models mo
    JOIN makes ma ON mo.make_id = ma.make_id
    WHERE mo.make_name IN ({g})
 
    UNION ALL
 
    SELECT
        mo.make_name,
        'Japanese' AS origin,
        mo.model_name,
        mo.year
    FROM models mo
    JOIN makes ma ON mo.make_id = ma.make_id
    WHERE mo.make_name IN ({j})
 
    ORDER BY origin, make_name, year, model_name
""".format(
    g=",".join("?" * len(german)),
    j=",".join("?" * len(japanese))
)
 
df_gvj = pd.read_sql_query(gvj_query, conn, params=german + japanese)
df_gvj.to_csv(os.path.join(OUTPUT_DIR, "german_vs_japanese.csv"), index=False)
print(f"  german_vs_japanese.csv: {len(df_gvj)} rows")
 
conn.close()
print("\nAll done! Files saved to:", OUTPUT_DIR)