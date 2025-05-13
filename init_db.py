import sqlite3, pathlib, importlib.resources

db = sqlite3.connect("qruns.db")
schema = pathlib.Path("init_runs.sql").read_text()
db.executescript(schema)
db.close()