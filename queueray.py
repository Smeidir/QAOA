# queue.py
import sqlite3, json, time, socket
from contextlib import closing
import ray



@ray.remote
class RunQueue:
    def __init__(self, db_path="qruns.db"):
        self.db_path = db_path

    def _conn(self):
        return sqlite3.connect(self.db_path, timeout=30,
                               isolation_level="IMMEDIATE")

    def next_job(self):
        """Atomically claim the next pending run (or return None)."""
        with closing(self._conn()) as db:
            cur = db.cursor()
            cur.execute("""
              UPDATE runs SET state='running', started_at=CURRENT_TIMESTAMP
              WHERE id = (
                SELECT id FROM runs
                WHERE state='pending'
                ORDER BY
                  json_extract(params, '$.depth') DESC,
                  json_extract(params, '$.graph_path') DESC
                LIMIT 1
              )
              RETURNING id, params
            """)
            row = cur.fetchone()
            db.commit()
            if row is None:
                return None
            run_id, params_json = row
            return run_id, json.loads(params_json)

    def mark_done(self,run_node, run_id, results_json):
        with closing(self._conn()) as db:
            db.execute("""
              UPDATE runs
              SET state='done', finished_at=CURRENT_TIMESTAMP, node=?,artefact_path=?
              WHERE id=?""", (run_node, results_json, run_id))
            db.commit()

    def mark_error(self, run_id, msg, retry=True):
        new_state = "pending" if retry else "error"
        with closing(self._conn()) as db:
            db.execute("""
              UPDATE runs
              SET state=?, error_msg=?
              WHERE id=?""", (new_state, msg[:255], run_id))
            db.commit()
