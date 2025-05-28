# worker.py  ────────────────────────────────────────────────────────────
import ray, json, pickle, pathlib, pandas as pd, time, socket
from pathlib import Path
from src.qaoa.core.QAOA import QAOArunner
import json_tricks
from src.qaoa.models import params

@ray.remote(num_gpus=8)
class Runner:
    """
    One Ray actor = one worker process that repeatedly:
       1. asks the queue for the next pending row
       2. un-pickles the graph
       3. runs QAOA   (build_circuit → run → to_dict)
       4. writes a one-row CSV + tells the queue "done"
    """

    def __init__(self, queue, outroot="/scratch/qresults"):
        self.queue   = queue

    # ────────────────────────────────────────────────────────────────
    #  public entry-point (called once from driver.py)
    # ────────────────────────────────────────────────────────────────
    def run_forever(self):
        while True:
            job = ray.get(self.queue.next_job.remote())
            if job is None:                       # queue empty
                break
            run_id, cfg = job
            results_json = self._do_run(run_id, cfg)
            if results_json:
                self.queue.mark_done.remote(socket.gethostname(),run_id, results_json)

    # ────────────────────────────────────────────────────────────────
    #  single-job execution
    # ────────────────────────────────────────────────────────────────
    def _do_run(self, run_id: int, cfg: dict):
        """
        cfg comes straight from qruns.db → params JSON.
        Expected keys  (by convention; tweak to taste):

            graph_path      str   absolute / relative .pkl file
            graph_label     str   human-friendly name (optional)
            seed            int   RNG seed (unused here but saved)
            …any QAOA-specific kwargs…

        Everything except graph_path / graph_label / seed is passed into
        QAOArunner(**kwargs).
        """
        graph_file  = pathlib.Path(cfg.pop("graph_path"))
        graph_label = cfg.pop("graph_label", graph_file.stem)

        # ---- output folder for this run --------------------------------

        # ---- load the graph (retworkx pickle) --------------------------
        with graph_file.open("rb") as f:
            graph_obj = pickle.load(f)

        # ---- build & run QAOA -----------------------------------------
        qaoa = QAOArunner(graph_obj, **cfg)   # cfg now only has QAOA kwargs
        qaoa.build_circuit()
        qaoa.run()

        # ---- collect results ------------------------------------------
        result_dict = qaoa.to_dict()
        result_dict.update({
            "graph_label": graph_label,
            **cfg          # echoes hyper-params for downstream analysis
        })

        return json_tricks.dumps(result_dict)    # ← return JSON string
