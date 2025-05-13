# export_db.py ──────────────────────────────────────────────────────
import sqlite3
import pandas as pd

# Connect to the database
db_path = "qruns.db"
conn = sqlite3.connect(db_path)

# Load all rows into a DataFrame
df = pd.read_sql_query("SELECT * FROM runs", conn)
conn.close()

# Save to CSV
csv_path = "runs_export.csv"
df.to_csv(csv_path, index=False)

print(f"✅ Exported {len(df)} runs to {csv_path}")
