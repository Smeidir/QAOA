

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from src.qaoa.models import params
def get_backend(mode, amount_shots=5000, verbose=False):
    match mode:
        case 'statevector':
            backend = AerSimulator(method="statevector", device='gpu')
    
            if verbose:
                print("You are running on the local ",print(backend.configuration()))

        case 'noisy_sampling':
            backend = AerSimulator.from_backend(FakeMarrakesh(), device='gpu')

            if verbose:
                print("Running on: AerSimulator with noise.", print(backend.configuration().to_dict()))

        case 'quantum_backend':
            QiskitRuntimeService.save_account(channel="ibm_quantum", token=params.api_key, overwrite=True, set_as_default=True)
            service = QiskitRuntimeService(channel='ibm_quantum')
            backend = service.least_busy(min_num_qubits=127)
            if verbose:
                print("Running on IBM quantum backend:", backend)
    
    return backend  # Added missing return statement


        