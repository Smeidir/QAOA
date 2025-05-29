from abc import ABC, abstractmethod
from scipy.optimize import minimize
import numpy as np
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
maxiter = 5000

class QAOAOptimizerStrategy(ABC):
    def __init__(self, optimizer, tol):
        self.optimizer = optimizer
        self.tol = tol


    @abstractmethod
    def minimize(self, init_params, *args):
        pass


class StatevectorOptimizer(QAOAOptimizerStrategy):
    def __init__(self, optimizer, tol, backend):
        super().__init__(optimizer, tol)
        self.backend = backend
    
    def minimize(self, init_params, circuit, hamiltonian):
        def cost_func(params):
            sv = Statevector.from_instruction(circuit.assign_parameters(params))
            cost = np.real(sv.expectation_value(hamiltonian))

            return cost

        return minimize(cost_func, init_params, method=self.optimizer,
                        tol=self.tol, options={'maxiter': 134})



class EstimatorOptimizer(QAOAOptimizerStrategy):
    def __init__(self, optimizer, tol, backend, shots, mitigation_fn=None):
        super().__init__(optimizer, tol)
        
        self.estimator = Estimator.from_backend(backend=backend)
        self.estimator.options.default_shots = shots
        if mitigation_fn:
            mitigation_fn(self.estimator)

    def minimize(self, init_params, circuit, hamiltonian):
        isa_hamiltonian = hamiltonian.apply_layout(circuit.layout)

        def cost_func(params):
            pub = (circuit, isa_hamiltonian, params)
            job = self.estimator.run([pub])
            result = job.result()
            cost = result[0].data.evs

            return cost

        return minimize(cost_func, init_params, method=self.optimizer,
                        tol=self.tol, options={'maxiter': maxiter})
    
    
class NoOptimizerStrategy(QAOAOptimizerStrategy):
    def __init__(self, mode="statevector", backend=None, shots=1024):
        super().__init__(optimizer=None, tol=None)
        self.mode = mode
        self.backend = backend
        self.shots = shots
    def minimize(self, *args, **kwargs):
        raise NotImplementedError("NoOptimizerStrategy does not support classical optimization.")
    
    def evaluate(self, params, circuit, hamiltonian):
        if self.mode == "statevector":
            sv = Statevector.from_instruction(circuit.assign_parameters(params))
            return np.real(sv.expectation_value(hamiltonian))


        elif self.mode in {"noisy_sampling", "quantum_backend"}:
            
            estimator = Estimator(mode=self.backend)
            estimator.options.default_shots = self.shots
            isa_hamiltonian = hamiltonian.apply_layout(circuit.layout)
            pub = (circuit, isa_hamiltonian, params)
            job = estimator.run([pub])
            return job.result()[0].data.evs.item()

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        
    def get_bitstring_probabilities(self, params, circuit):
        n = circuit.num_qubits
        

        if self.mode == "statevector":
            
            state = Statevector.from_instruction(circuit.assign_parameters(params))
            probs = {int(k, 2): v for k, v in state.probabilities_dict().items()}
            return probs


        elif self.mode in {"noisy_sampling", "quantum_backend"}:
            circuit = circuit.assign_parameters(params)
            pub = (circuit)
            sampler = Sampler(mode=self.backend)
            sampler.options.default_shots = self.shots
            if self.mode == 'quantum_backend':
                self.set_error_mitigation(sampler)
            job = sampler.run([pub])
            counts_int = job.result()[0].data.meas.get_int_counts()
            shots = sum(counts_int.values())
            final_distribution_int = {key: val/shots for key, val in counts_int.items()}
            return final_distribution_int

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
