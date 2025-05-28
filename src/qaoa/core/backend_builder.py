
import os
# Must be at top, before importing Aer
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"


from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from src.qaoa.models import params
def get_backend(mode, amount_shots=5000, verbose=False):
    match mode:
        case 'statevector':
            backend = AerSimulator(method="statevector", max_parallel_experiments=50)
    
            if verbose:
                print("You are running on the local ",print(backend.configuration()))

        case 'noisy_sampling':
            backend = AerSimulator.from_backend(FakeMarrakesh(), max_parallel_experiments=50)

            if verbose:
                print("Running on: AerSimulator with noise.", print(backend.configuration().to_dict()))

        case 'quantum_backend':
            QiskitRuntimeService.save_account(channel="ibm_quantum", token=params.api_key, overwrite=True, set_as_default=True)
            service = QiskitRuntimeService(channel='ibm_quantum')
            backend = service.least_busy(min_num_qubits=127)
            if verbose:
                print("Running on IBM quantum backend:", backend)
    
    return backend  # Added missing return statement


        