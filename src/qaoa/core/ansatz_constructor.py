from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp


def build_ansatz(mode, cost_hamiltonian, qubits, depth , warm_start_seed = None):
    match mode:
        case 'vanilla':
            return _build_vanilla_ansatz(cost_hamiltonian, qubits,depth, warm_start_seed)
        case 'multiangle':
            return _build_multiangle_ansatz(cost_hamiltonian, qubits,depth, warm_start_seed)
       # case 'controlled':
       #    return _build_constrained_ansatz(cost_hamiltonian, qubits, depth, warm_start_seed)

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
import numpy as np

def _build_vanilla_ansatz(cost_hamiltonian: SparsePauliOp, num_qubits: int, depth: int,
                          warm_start_seed=None) -> QuantumCircuit:
    if warm_start_seed is None:
        return QAOAAnsatz(cost_operator=cost_hamiltonian, reps=depth, flatten=True)

    thetas = warm_start_seed
    initial_state = QuantumCircuit(num_qubits)
    for q, theta in enumerate(thetas):
        initial_state.ry(theta, q)

    mixer = QuantumCircuit(num_qubits)
    beta = Parameter("β")
    for q, theta in enumerate(thetas):
        mixer.ry(-theta, q)
        mixer.rz(2 * beta, q)
        mixer.ry(theta, q)

    return QAOAAnsatz(cost_operator=cost_hamiltonian, mixer_operator=mixer,
                      initial_state=initial_state, reps=depth, flatten=True)

def _build_multiangle_ansatz(cost_hamiltonian: SparsePauliOp, num_qubits: int, depth: int,
                             warm_start_seed=None) -> QuantumCircuit:
    cost_terms = list(zip([str(p) for p in cost_hamiltonian.paulis], cost_hamiltonian.coeffs))
    qc = QuantumCircuit(num_qubits)

    if warm_start_seed is not None:
        for i, theta in enumerate(warm_start_seed):
            qc.ry(theta, i)
    else:
        for i in range(num_qubits):
            qc.h(i)

    for d in range(depth):
        gamma_params = [Parameter(f"γ_{d}_{i}") for i in range(len(cost_terms))]
        beta_params = [Parameter(f"β_{d}_{i}") for i in range(num_qubits)]

        for i, (pauli, coeff) in enumerate(cost_terms):
            z_qubits = [q for q, p in enumerate(pauli) if p == "Z"]
            if len(z_qubits) == 1:
                qc.rz(2 * coeff * gamma_params[i], z_qubits[0])
            elif len(z_qubits) == 2:
                qc.cx(z_qubits[0], z_qubits[1])
                qc.rz(2 * gamma_params[i], z_qubits[1])
                qc.cx(z_qubits[0], z_qubits[1])
            else:
                raise ValueError(f"Unsupported term with >2 Zs: {pauli}")

        for i in range(num_qubits):
            if warm_start_seed is not None:
                qc.ry(-warm_start_seed[i], i)
            qc.rx(2 * beta_params[i], i)
            if warm_start_seed is not None:
                qc.ry(warm_start_seed[i], i)
    return qc

"""      
def _build_constrained_ansatz(cost_hamiltonian: SparsePauliOp, num_qubits: int, depth: int,
                          warm_start_seed=None) -> QuantumCircuit:
    if warm_start_seed is None:
        return QAOAAnsatz(cost_operator=cost_hamiltonian, reps=depth, flatten=True)

    thetas = warm_start_seed
    initial_state = QuantumCircuit(num_qubits)
    if warm_start_seed:
        for i, theta in enumerate(warm_start_seed):
            qc.ry(theta, i)
    else:
        for i in range(num_qubits):
            qc.h(i)

    mixer = QuantumCircuit(num_qubits)
    beta = Parameter("β")

    #go through the neighbourhood/ edges from a given vertex
    # for all in the neighbourhood, construct a CNOT with control the neighbour, target ancilla qubit
    # one ancilla qubit for each
    # make a controlled NOT gate from all the ancilla qubits and to the given vertex
    # uncompute the ancillas. These can be reused
    # probably needs a lot of ancillas?
    for q, theta in enumerate(thetas):
        mixer.ry(theta, q)
        mixer.rz(2 * beta, q)
        mixer.ry(-theta, q)

    return QAOAAnsatz(cost_operator=cost_hamiltonian, mixer_operator=mixer,
                      initial_state=initial_state, reps=depth, flatten=True)
"""