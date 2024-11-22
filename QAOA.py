import time
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
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate
from qiskit.circuit import Parameter
from qiskit_optimization.translators import from_docplex_mp, to_ising
import rustworkx as rx
from qiskit_optimization.converters import QuadraticProgramToQubo


from solver import Solver
from qiskit_ibm_runtime import QiskitRuntimeService

class QAOArunner():
    """
    A class with all the functionality needed to create quantum circuit and run using the QAOA algorithm.
    inputs:
    Simulation: boolean, whether to run locally or on IBM cloud
    Graph: pygraph, the problem to solve
    initialization: string, the method of initializing the weights
    optimizer: what scipy optimizer to use.
    """
    def __init__(self, graph, simulation=True, param_initialization="uniform",optimizer="COBYLA", qaoa_variant ='normal', solver = None, warm_start=False, test_hamil = False):
        self.graph = graph
        self.simulation = simulation
        self.param_initialization = param_initialization
        self.qaoa_variant = qaoa_variant
        self.optimizer = optimizer
        self.solution = None
        self.solver = solver
        self.warm_start =warm_start
        self.test_hamil = test_hamil


        #TODO: calculate num_qubits. For now, it's just the amount of nodes
        self.num_qubits = len(self.graph.nodes())
        

    def build_circuit(self):
        """ 
        Convert graph to pauli list and then to a cost hamiltonian, and converts this into a circuit.
        Pauli lists are operation instructions for the quantum circuit, and are 
        strings with mostly I-s and some Z-s. 
        They represent Z-operations on some qubits and I-operations on others.
        Cost hamiltonian is the way the cirucit understands costs (?)
        updates self.: backend, circuit, cost_hamiltonian
        """
        
        pauli_list = []
        for edge in list(self.graph.edge_list()):
            paulis = ["I"]*len(self.graph)
            paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

            weight = self.graph.get_edge_data(edge[0],edge[1])
            pauli_list.append(("".join(paulis)[::-1], weight))
        

        conv = QuadraticProgramToQubo()
        solver = Solver(self.graph, relaxed = False) #use solver not to solve, but to get the qubo formulation - must not be relaxd!
        cost_hamiltonian = to_ising(conv.convert(solver.get_qp()))

        cost_hamiltonian_tuples = [(pauli, coeff) for pauli, coeff in zip([str(x) for x in cost_hamiltonian[0].paulis], cost_hamiltonian[0].coeffs)]
        cost_hamiltonian = SparsePauliOp.from_list(pauli_list)

        print('Should be: ', cost_hamiltonian)

        if self.test_hamil: 
            cost_hamiltonian = SparsePauliOp.from_list(cost_hamiltonian_tuples)
            print('Is: ', cost_hamiltonian)
        print('Num qubits: ', cost_hamiltonian.num_qubits)
        qc = None
        if self.qaoa_variant =='normal':
            if self.warm_start:
                solver = Solver(self.graph, relaxed = True)
                solution_values,_ = solver.solve() #solving this twice, not necessarily good. runs fast though
                initial_state = QuantumCircuit(self.num_qubits)
                thetas = 2*np.arcsin(np.sqrt(solution_values))
                for qubit in range(self.num_qubits): #TODO: must check if they are the correct indices - qubits on IBM might be opposite ordering
                    initial_state.ry(thetas[qubit],qubit)
                mixer_state = QuantumCircuit(self.num_qubits)
                mixer_param = Parameter('β')
                for qubit in range(self.num_qubits):
                    mixer_state.ry(thetas[qubit],qubit) 
                    mixer_state.rz(mixer_param, qubit)# Assign a placeholder beta parameter 
                    mixer_state.ry(-thetas[qubit],qubit)
                qc = QAOAAnsatz(cost_operator = cost_hamiltonian, reps = params.depth, initial_state=initial_state, mixer_operator=mixer_state)

            else:
                qc = QAOAAnsatz(cost_operator = cost_hamiltonian, reps = params.depth)
            qc.measure_all()
        elif self.qaoa_variant =='multiangle': 
            multiangle_gammas = [[Parameter(f'γ_{l}_{i}') for i in range(len(self.graph.edges()))] for l in range(params.depth)]
            multiangle_betas = [[Parameter(f'β_{l}_{i}') for i in range(self.num_qubits)] for l in range(params.depth)]
    
            qc = QuantumCircuit(self.num_qubits)
            for _ in range(self.num_qubits): 
                qc.rz(np.pi/2, _) #not hadamards, but close. The same as qiskit uses
                qc.sx(_)
                qc.rz(np.pi/2, _)

            for i in range(params.depth):
                for idx, edge in enumerate(self.graph.edge_list()):
                    qc.cx(edge[0], edge[1])
                    qc.rz(multiangle_gammas[i][idx], edge[1])
                    qc.cx(edge[0], edge[1])
                for idx in range(self.num_qubits):#TODO: add multiangle here
                    qc.rx(2*multiangle_betas[i][idx], idx)
            qc.measure_all()

        self.build_backend()
        pm = generate_preset_pass_manager(optimization_level=2,backend=self.backend)
        candidate_circuit = pm.run(qc)
        self.circuit = candidate_circuit
        self.cost_hamiltonian = cost_hamiltonian

        cost_operator = candidate_circuit.cost_operator
        mixer_operator = candidate_circuit.mixer_operator


        commutator = cost_operator @ mixer_operator - mixer_operator @ cost_operator

        print("commutator: ", commutator)

 
    def build_backend(self):

        if self.simulation: 
            self.backend = GenericBackendV2(num_qubits=len(self.graph)) #TODO: make this generic for k-cut
        else:

            QiskitRuntimeService.save_account(channel="ibm_quantum", token=params.api_key, overwrite=True, set_as_default=True)
            service = QiskitRuntimeService(channel='ibm_quantum')
            self.backend = service.least_busy(min_num_qubits=127)
            print(self.backend)
            #raise NotImplementedError('Running on IBM not implemented yet. Set Simulation to True instead to run locally.')
        
    def draw_circuit(self):
        self.circuit.draw('mpl', fold=False, idle_wires=False)

    def get_init_params(self): #TODO: add support for multiangle
        supported_params = ['gaussian','uniform','machine_learning'] #TODO: move into init

        if self.param_initialization not in supported_params:
            raise ValueError(f'Non-supported param initializer. Your param: {self.param_initialization} not in supported parameters:{supported_params}.')

        param_length = None #none so if its not changed its easier to see bugs - if it was 0 might be bugs furhter down the line
        if self.qaoa_variant == "normal":
            param_length = 2
        elif self.qaoa_variant == "multiangle":
            param_length = self.num_qubits + len(self.graph.edges())

        match self.param_initialization: 
            case 'uniform':
                param_length = param_length*params.depth
                init_params = np.random.uniform(0,np.pi,param_length)
                print(f'Init_params, for error checking: {init_params}')
                return init_params
            case 'gaussian':
                param_length = param_length*params.depth
                init_params = np.random.normal(0,np.pi,param_length)
                print(f'Init_params, for error checking: {init_params}')
                return init_params

            case 'machine_learning':
                raise NotImplementedError('Machine Learning not implemented yet. Use uniform or gaussian instead.') 



    def run(self):
        self.objective_func_vals = []
        init_params = self.get_init_params()

        start_time = time.time()
        with Session(backend = self.backend) as session:
                estimator = Estimator(mode=session)
                estimator.options.default_shots = 1000
                if not self.simulation:
                        # Set simple error suppression/mitigation options
                        estimator.options.dynamical_decoupling.enable = True
                        estimator.options.dynamical_decoupling.sequence_type = "XY4"
                        estimator.options.twirling.enable_gates = True
                        estimator.options.twirling.num_randomizations = "auto"
                start_time = time.time()
                result = minimize(
                self.cost_func_estimator, 
                init_params,
                args= (self.circuit, self.cost_hamiltonian, estimator),
                method = self.optimizer,
                tol = 1e-2
                )
                end_time = time.time()
        elapsed_time = end_time-start_time
        print('Elapsed time:',elapsed_time)        
        print(result)
        self.result = result
        self.circuit = self.circuit.assign_parameters(self.result.x)
        self.solution = self.calculate_solution()
        self.objective_value = self.evaluate_sample()
        

    def evaluate_sample(self) -> float:
        assert len(self.solution) == len(list(self.graph.nodes())), "The length of x must coincide with the number of nodes in the graph."
        return sum(self.solution[u] * (1 - self.solution[v]) + self.solution[v] * (1 - self.solution[u]) for u, v in set(self.graph.edge_list()))




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
    def plot_result(self):
        colors = ["tab:grey" if i == 0 else "tab:purple" for i in self.solution]
        pos, default_axes = rx.spring_layout(self.graph), plt.axes(frameon=True)
        rx.visualization.mpl_draw(self.graph, node_color=colors, node_size=100, alpha=0.8, pos=pos)


    def get_prob_distribution(self):
        """
        Gives the probability distribution per possible outcome.
        Must be called after run().
        Prints the results.
        TODO: make better?
        """


        pub = (self.circuit,)
        sampler = Sampler(mode=self.backend)
        sampler.options.default_shots=1000

        if not self.simulation:
                # Set simple error suppression/mitigation options
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.num_randomizations = "auto"

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
        most_likely_bitstring = to_bitstring(most_likely, len(self.graph))#TODO: change to amount of qubits
        most_likely_bitstring.reverse()

        print("Result bitstring:", most_likely_bitstring)

    def calculate_solution(self): #TODO: må da finnes en lettere måte?
        #TODO: support fior å finne flere av de mest sannsynlige?
        
        pub = (self.circuit,)
        sampler = Sampler(mode=self.backend)
        sampler.options.default_shots=1000

        if not self.simulation:
        # Set simple error suppression/mitigation options
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.num_randomizations = "auto"

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
        most_likely_bitstring = to_bitstring(most_likely,len(self.graph))#TODO: change to amount of qubits
        most_likely_bitstring.reverse()

        return most_likely_bitstring

    def compare_solutions(self, classic_solution):
        if not self.solution:
            raise ReferenceError("Solution not initalized yet. run()-function must be called before solution can be generated.")
        assert len(self.solution) == len(classic_solution[0]), 'Solutions not the same length.' #TODO: error relating to length of qubits for kcut which requires more qubits
        bools = [a==b for a,b in zip(classic_solution[0],self.solution)]
        bools_reversed =[a!=b for a,b in zip(classic_solution[0],self.solution)]
        print("Result quantum", self.solution, "Objective value: ", self.objective_value)
        print("Result input (classical)", classic_solution[0], "Objective Value: ", classic_solution[1])
        print("Same solution", all(bools) or all(bools_reversed)) #same cut but different partitions
        print("Same objective function value: ", classic_solution[1] == self.objective_value)

