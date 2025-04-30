from qiskit_aer import AerSimulator, StatevectorSimulator, Aer
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeBrisbane,FakeMelbourneV2

from qaoa.models import params

def get_backend(mode, amount_shots=5000, verbose=False):
    match mode:
        case 'statevector':
            backend = AerSimulator(method="statevector",max_parallel_threads=4,
    max_parallel_experiments=1,
    max_parallel_shots=4)
    
            if verbose:
                print("You are running on the local ",print(backend.configuration()))

        case 'noisy_sampling':
            service = QiskitRuntimeService()
            print(service.backends())
            print(FakeMelbourneV2())
            backend = service.backend("ibm_brisbane")
            noise_model = NoiseModel.from_backend(backend)
            backend = AerSimulator.from_backend(noise_model = noise_model,max_parallel_threads=16,
    max_parallel_experiments=1,
    max_parallel_shots=16)
            if verbose:
                print("Running on: AerSimulator with noise.", print(backend.configuration().to_dict()))

        case 'quantum_backend':
            QiskitRuntimeService.save_account(channel="ibm_quantum", token=params.api_key, overwrite=True, set_as_default=True)
            service = QiskitRuntimeService(channel='ibm_quantum')
            backend = service.least_busy(min_num_qubits=127)
            if verbose:
                print("Running on IBM quantum backend:", backend)
    
    return backend  # Added missing return statement

get_backend('noisy_sampling', verbose=True)
        