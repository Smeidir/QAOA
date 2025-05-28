#!/usr/bin/env python
"""
benchmark_dm.py – quick-and-dirty timing of Qiskit-Aer simulator settings

Usage examples
--------------
# default: qubit counts [6, 8, 10, 12], 5000 shots each
python benchmark_dm.py

# specify your own list of qubit numbers
python benchmark_dm.py --qubits 6 9 12 15 --shots 10000 --csv timings.csv
"""
import argparse, time, csv, os, sys, warnings
from collections import namedtuple

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import depolarizing_error

Result = namedtuple("Result", "qubits method backend shots elapsed")

# ---------------------------------------------------------------------
# helper: build a trivial depth-4 circuit (RX, CZ, RZ, measure all)
# ---------------------------------------------------------------------
def make_dummy_circuit(nqubits: int, shots: int):
    qc = QuantumCircuit(nqubits, nqubits, name=f"dummy_{nqubits}q")
    # Layer 1 – single-qubit rotations
    for q in range(nqubits):
        qc.rx(0.17 * (q+1), q)
    # Layer 2 – CZ chain
    for q in range(0, nqubits-1, 2):
        qc.cz(q, q+1)
    # Layer 3 – more rotations
    for q in range(nqubits):
        qc.rz(0.11 * (q+1), q)
    qc.barrier()
    qc.measure(range(nqubits), range(nqubits))
    return transpile(qc, optimization_level=0)

# ---------------------------------------------------------------------
# helper: very light noise model (depolarizing) so we still need DM
# ---------------------------------------------------------------------
def simple_noise_model(nqubits):
    nm = NoiseModel()
    # single-qubit depolarizing 1% / two-qubit 3 %
    dep1 = depolarizing_error(0.01, 1)
    dep2 = depolarizing_error(0.03, 2)
    for q in range(nqubits):
        nm.add_quantum_error(dep1, ["rx", "rz"], [q])
    for q in range(nqubits-1):
        nm.add_quantum_error(dep2, ["cz"], [q, q+1])
    return nm

# ---------------------------------------------------------------------
def build_backend(method: str, shots: int, noise=None):
    device = "GPU" if "gpu" in method else "CPU"
    cfg = dict(method=method, device=device, shots=shots)
    if "gpu" in method:               # silence harmless CPU fall-back warning
        warnings.filterwarnings("ignore", category=UserWarning,
                                message=".*no GPU .* switching to CPU.*")
        cfg["cu_statevec"] = True
    return AerSimulator(**cfg, noise_model=noise)

# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", nargs="+", type=int, default=[6, 8, 10, 12],
                    help="space-separated list of qubit counts")
    ap.add_argument("--shots", type=int, default=5000)
    ap.add_argument("--csv",   help="output CSV file")
    args = ap.parse_args()

    # identify whether a CUDA device is visible
    cuda_visible = bool(os.getenv("CUDA_VISIBLE_DEVICES"))
    if not cuda_visible:
        print("[Info] CUDA_VISIBLE_DEVICES not set – GPU benchmarks will be skipped",
              file=sys.stderr)

    all_results = []
    for nq in args.qubits:
        circ  = make_dummy_circuit(nq, args.shots)
        noise = simple_noise_model(nq)

        # ---------- density_matrix on CPU ----------
        sim_cpu = build_backend("density_matrix", args.shots, noise)
        t0 = time.perf_counter()
        sim_cpu.run(circ).result()
        dt = time.perf_counter() - t0
        all_results.append(Result(nq, "density_matrix", "CPU", args.shots, dt))

        # ---------- density_matrix on GPU ----------
        if cuda_visible:
            sim_gpu = build_backend("density_matrix_gpu", args.shots, noise)
            t0 = time.perf_counter()
            sim_gpu.run(circ).result()
            dt = time.perf_counter() - t0
            all_results.append(Result(nq, "density_matrix_gpu", "GPU", args.shots, dt))

        # ---------- statevector baseline ----------
        sv = build_backend("statevector", args.shots, noise=None)  # no noise
        t0 = time.perf_counter()
        sv.run(circ).result()
        dt = time.perf_counter() - t0
        all_results.append(Result(nq, "statevector", "CPU/GPU", args.shots, dt))

    # ------------------------------------------------------------------
    # pretty-print table
    # ------------------------------------------------------------------
    hdr = f"{'qubits':>6}  {'method':18}  {'device':6}  {'shots':>5}  {'sec':>8}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in all_results:
        print(f"{r.qubits:6d}  {r.method:18}  {r.backend:6}  {r.shots:5d}  {r.elapsed:8.3f}")

    # save CSV if requested
    if args.csv:
        with open(args.csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(Result._fields)
            w.writerows(all_results)
        print(f"\nSaved timings to {args.csv}")

if __name__ == "__main__":
    main()
