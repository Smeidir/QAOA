import sqlite3, pathlib, importlib.resources

# Check if database file exists and remove it
db_path = pathlib.Path("qruns.db")
if db_path.exists():
    db_path.unlink()  # Delete the file

# Create a new database
db = sqlite3.connect("qruns.db")
schema = pathlib.Path("init_runs.sql").read_text()
db.executescript(schema)
db.close()