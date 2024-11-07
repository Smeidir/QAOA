from matplotlib import pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import QAOAAnsatz
import params
import numpy as np
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from scipy.optimize import minimize
from qiskit_ibm_runtime import SamplerV2 as Sampler

class QAOArunner():
    """
    A class with all the functionality needed to create quantum circuit and run using the QAOA alogrithm.
    inputs:
    Simulation: boolean, whether to run locally or on IBM cloud
    Graph: pygraph, the problem to solve
    initialization: string, the method of initializing the weights
    optimizer: what scipy optimizer to use.
    """
    def __init__(self, graph, simulation=True, initialization="normal",optimizer="COBYLA"¨, multiangle =False):
        self.graph = graph
        self.simulation = simulation
        self.initialization = initialization
        self.optimizer = optimizer
        

    def build_circuit(self):
        """ 
        Convert graph to pauli list and then to a cost hamiltonian, and converts this into a circuit.
        Pauli lists are operation instructions for the quantum circuit, and are 
        strings with mostly I-s and some Z-s. 
        They represent Z-operations on some qubits and I-operations on others.
        Cost hamiltonian is the way the cirucit understands costs (?)
        Returns the cost hamiltonian
        """

        pauli_list = []
        for edge in list(self.graph.edge_list()):
            paulis = ["I"]*len(self.graph)
            paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

            weight = self.graph.get_edge_data(edge[0],edge[1])
            pauli_list.append(("".join(paulis)[::-1], weight))
        cost_hamiltonian = SparsePauliOp.from_list(pauli_list)

        circuit = QAOAAnsatz(cost_operator = cost_hamiltonian, reps = params.depth)
        circuit.measure_all()


        self.build_backend()
        pm = generate_preset_pass_manager(optimization_level=2,backend=self.backend)

        candidate_circuit = pm.run(circuit)
        self.circuit = candidate_circuit
        self.cost_hamiltonian = cost_hamiltonian

        
    
    def build_backend(self):

        if self.simulation: 
            self.backend = GenericBackendV2(num_qubits=params.graph_size) #TODO: make this generic for k-cut
        else:
            raise NotImplementedError('Running on IBM not implemented yet. Set Simulation to True instead to run locally.')
        
    def draw_circuit(self):
        self.circuit.draw('mpl', fold=False, idle_wires=False)

    def get_init_params(self): #TODO: add support for multiangle

        match self.initialization: 
            case 'warm_start': 
                raise NotImplementedError('Not Implemented yet. Use normal instead.') 
            case 'normal':
                initial_gamma = np.pi
                initial_beta = np.pi/2 #todo change 
                init_params = [(initial_gamma, initial_beta) for _ in range(params.depth)]

                init_params = [number for tup in init_params for number in tup]
                return init_params

    def run(self):
        self.objective_func_vals = []
        init_params = self.get_init_params()


        with Session(backend = self.backend) as session:
                estimator = Estimator(mode=session)
                estimator.options.default_shots = 1000
    
                result = minimize(
                self.cost_func_estimator, 
                init_params,
                args= (self.circuit, self.cost_hamiltonian, estimator),
                method = self.optimizer,
                tol = 1e-2
            )
        print(result)
        self.result = result


    def cost_func_estimator(self,params, ansatz, hamiltonian, estimator):
        #TODO: see if this can be optimized
        #transform observable defined on virtual qubits to an observable defined on all physical qubits

        isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

        pub = (ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])

        results = job.result()[0]
        cost = results.data.evs

        self.objective_func_vals.append(cost)

        return cost

    def draw_objective_value(self):
        """
        Draws the objective value function evolution over time.
        Must be called after run()
        """


        plt.figure(figsize=(12,6))
        plt.plot(self.objective_func_vals)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()

    def get_prob_distribution(self):
        """
        Gives the probability distribution per possible outcome.
        Must be called after run().
        Prints the results.
        TODO: make better?
        """

        optimized_circuit = self.circuit.assign_parameters(self.result.x)
        pub = (optimized_circuit,)
        sampler = Sampler(mode=self.backend)
        sampler.options.default_shots=10000

        job = sampler.run([pub], shots=int(1e4))
        counts_int = job.result()[0].data.meas.get_int_counts()
        print(counts_int)
        counts_bin = job.result()[0].data.meas.get_counts()
        shots = sum(counts_int.values())
        final_distribution_int = {key: val/shots for key, val in counts_int.items()}
        final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}
        print(final_distribution_int)
        def to_bitstring(integer, num_bits):
            result = np.binary_repr(integer, width=num_bits)
            return [int(digit) for digit in result]

        keys = list(final_distribution_int.keys())
        values = list(final_distribution_int.values())

        most_likely = keys[np.argmax(np.abs(values))]
        most_likely_bitstring = to_bitstring(most_likely, params.graph_size)#TODO: change to amount of qubits
        most_likely_bitstring.reverse()

        print("Result bitstring:", most_likely_bitstring)
