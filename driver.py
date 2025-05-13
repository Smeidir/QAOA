# driver.py ───────────────────────────────────────────────────────────
import os, time, threading, sqlite3

import ray
from tqdm import tqdm

from queueray import RunQueue
from worker   import Runner

# ---------------------------------------------------------------------
# 1.  Connect to (or start) Ray
# ---------------------------------------------------------------------
ray.init(address="auto")                # or ray.init() for single-node test

# ---------------------------------------------------------------------
# 2.  Spawn the central RunQueue actor (detached = survives driver exit)
# ---------------------------------------------------------------------
queue = RunQueue.options(
            name="runqueue",
            lifetime="detached").remote()

# ---------------------------------------------------------------------
# 3.  Launch worker actors and kick off their run loop
#     num_cpus per Runner is defined in worker.py (@ray.remote(num_cpus=…))
# ---------------------------------------------------------------------

cluster_cpus = int(ray.cluster_resources().get("CPU", 0))
cpus_per_worker = 4  # must match @ray.remote(num_cpus=...)

num_workers = cluster_cpus // cpus_per_worker

print(f"Launching {num_workers} workers ({cpus_per_worker} CPUs each) across {cluster_cpus} total CPUs")

workers = [Runner.remote(queue) for _ in range(num_workers)]

# ---------------------------------------------------------------------
# 4.  Progress bar (tqdm) that updates every 5 s
# ---------------------------------------------------------------------
with sqlite3.connect("qruns.db") as db:
    total_jobs = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

pbar = tqdm(total=total_jobs,
            desc="QAOA jobs finished",
            unit="job",
            dynamic_ncols=True)

def monitor():
    prev = 0
    while True:
        with sqlite3.connect("qruns.db") as db:
            done  = db.execute("SELECT COUNT(*) FROM runs "
                               "WHERE state='done'").fetchone()[0]
            error = db.execute("SELECT COUNT(*) FROM runs "
                               "WHERE state='error'").fetchone()[0]
            finished = done + error
        delta = finished - prev
        if delta > 0:
            pbar.update(delta)
            prev = finished
        time.sleep(5)

threading.Thread(target=monitor, daemon=True).start()

# ---------------------------------------------------------------------
# 5.  Wait for all Runner actors to exit (queue empty), then shut down
# ---------------------------------------------------------------------
ray.get([w.run_forever.remote() for w in workers])
pbar.close()
print("✅ All runs completed.")
ray.shutdown()
