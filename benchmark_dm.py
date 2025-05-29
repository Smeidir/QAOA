#!/usr/bin/env python
"""
benchmark_dm.py  –  quick timing of *one* MaxCut/QAOA circuit under many
                    Qiskit-Aer simulator configurations.

Edit the SIMS list below to add / remove settings.

Examples
--------
python benchmark_dm.py --shots 5000
python benchmark_dm.py --shots 10000 --csv timings.csv
"""

import argparse, csv, os, sys, time, warnings
from collections import namedtuple
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from src.qaoa.models.MaxCutProblem import MaxCutProblem
from src.qaoa.core.QAOA import QAOArunner

# ──────────────────────────── 1. EDIT HERE ────────────────────────────
# tag,     method,              device,   extra kwargs passed to AerSimulator()
SIMS = [("GPU-Baseline",  "density_matrix",  "GPU", {}),
("GPU-CuStateVec",  "density_matrix",  "GPU", {"cuStateVec_enable": True}),
("GPU-Blocking",  "density_matrix",  "GPU", {'blocking_enable':True, 'blocking_qubits': 9}),
("GPU-batchedShots",  "density_matrix",  "GPU", {"batched_shots_gpu": True}),
("GPU-ThreadsPerDevice8",  "density_matrix",  "GPU", {"num_threads_per_device ": 8}),
("GPU-ThreadsPerDevice32",  "density_matrix",  "GPU", {"num_threads_per_device ": 32}),
("GPU-ShotBranching",  "density_matrix",  "GPU", {"shot_branching_enable ": True}),
("GPU-ShotBranchingSampling",  "density_matrix",  "GPU", {"shot_branching_sampling_enable ": True}),
("GPU-TensorNet",  "tensor_network",  "GPU", {}),
("GPU-TensorNet 12 qubits",  "tensor_network",  "GPU", {'tensor_network_num_sampling_qubits':12}),

]

"""


    ("CPU-Baseline",  "density_matrix",      "CPU", {}),
    ("CPU-MaxShotSize",  "density_matrix",      "CPU", {'max_shot_size':625}),
    ("CPU-ParallelShot/Thread 64",  "density_matrix",      "CPU", {'max_parallel_threads': 64,'max_parallel_shots': 64 }),
    ("CPU-ParallelShot/Thread 1",  "density_matrix",      "CPU", {'max_parallel_threads': 1,'max_parallel_shots': 1 }),
    ("CPU-ParalellExperiments",  "density_matrix",      "CPU", {'max_parallel_experiments': 0}),
    ("CPU-Blocking",  "density_matrix",      "CPU", {'blocking_enable':True, 'blocking_qubits': 9}),
    ("CPU-ShotBranching",  "density_matrix",      "CPU", {'shot_branching_enable':True}),
    ("CPU-ShotBranchingSampling",  "density_matrix",      "CPU", {'shot_branching_sampling_enable': True}),

"""
# ───────────────────────────────────────────────────────────────────────

Result = namedtuple("Result", "tag method device shots sec n_qubits")

def build_test_circuit() -> "QuantumCircuit":
    """Return the single (largest) circuit you want to benchmark."""
    problem      = MaxCutProblem()
    big_graph    = problem.get_erdos_renyi_graphs_paper1()[2]  # paper1_2.pkl
    qaoa         = QAOArunner(big_graph, depth=1)
    qaoa.build_circuit()
    circ = qaoa.circuit.copy()
    param_dict = {param: 0.1 for param in circ.parameters}  # or random values, e.g. np.random.rand()
    bound_circ = circ.assign_parameters(param_dict, inplace=False)
    return bound_circ

def build_backend(method, device, shots, noise, extra):
    """Return a configured AerSimulator."""
    cfg = dict(method=method, device=device, shots=shots, noise_model=noise)
    cfg.update(extra)                      # pull in extra kwargs from SIMS
    if device == "GPU":
        warnings.filterwarnings("ignore",
            message=".*no GPU .* switching to CPU.*", category=UserWarning)
    return AerSimulator(**cfg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=5000)
    ap.add_argument("--csv",   help="file to write results (optional)")
    args = ap.parse_args()

    cuda_seen = bool(os.getenv("CUDA_VISIBLE_DEVICES"))
    if not cuda_seen:
        print("[Info] CUDA_VISIBLE_DEVICES empty – GPU sims will fall back to CPU.",
              file=sys.stderr)

    circ        = build_test_circuit()
    n_qubits    = circ.num_qubits
    noise_model = NoiseModel.from_backend(FakeMarrakesh())  # realistic IBM noise

    results = []
    for tag, method, device, extra in SIMS:
        if device == "GPU" and not cuda_seen:
            print(f"[Skip] {tag}: no GPU visible.")
            continue
        
        backend = build_backend(method, device, args.shots, noise_model, extra)
        t0      = time.perf_counter()
        backend.run(circ).result()
        dt      = time.perf_counter() - t0
        results.append(Result(tag, method, device, args.shots, dt, n_qubits))
        print(f"[Done] {tag:8s}  {dt:8.3f} s")

    # ─────────────── print pretty table ───────────────
    hdr = f"{'tag':25} {'method':18} {'dev':6} {'shots':5} {'qubits':6} {'sec':>8}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in results:
        print(f"{r.tag:8} {r.method:18} {r.device:6} {r.shots:5d} "
              f"{r.n_qubits:6d} {r.sec:8.3f}")

    # optional CSV
    if args.csv:
        with open(args.csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(Result._fields)
            w.writerows(results)
        print(f"\nSaved timings → {args.csv}")

if __name__ == "__main__":
    main()
