import time
from matplotlib import pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import QAOAAnsatz
import params
import numpy as np
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from scipy.optimize import minimize
from qiskit_algorithms.optimizers import COBYLA
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.primitives import BackendSampler
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate
from qiskit.circuit import Parameter
from qiskit_optimization.translators import from_docplex_mp, to_ising
from qiskit.primitives import BackendSampler
import rustworkx as rx
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import Operator
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizer)
from solver import Solver
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_algorithms import QAOA

class QAOArunner():
    """
    A class with all the functionality needed to create quantum circuit and run using the QAOA algorithm.
    inputs:
    Simulation: boolean, whether to run locally or on IBM cloud
    Graph: pygraph, the problem to solve
    initialization: string, the method of initializing the weights
    optimizer: what scipy optimizer to use.
    """
    def __init__(self, graph, simulation=True, param_initialization="uniform",optimizer="COBYLA", qaoa_variant ='vanilla', solver = None, warm_start=False,restrictions=False, k=2, test = False, flatten = True):
        
        if qaoa_variant not in params.supported_qaoa_variants:
            raise ValueError(f'Non-supported param initializer. Your param: {qaoa_variant} not in supported parameters:{params.supported_qaoa_variants}.')
        if param_initialization not in params.supported_param_inits:
            raise ValueError(f'Non-supported param initializer. Your param: {param_initialization} not in supported parameters:{params.supported_param_inits}.')
        if optimizer not in params.supported_optimizers:
            raise ValueError(f'Non-supported param initializer. Your param: {optimizer} not in supported parameters:{params.supported_optimizers}.')
        
        self.graph = graph
        self.simulation = simulation
        self.param_initialization = param_initialization
        self.qaoa_variant = qaoa_variant
        self.optimizer = optimizer
        self.solution = None
        self.solver = solver
        self.warm_start =warm_start
        self.restrictions = restrictions
        self.test = test
        self.flatten = flatten

        self.num_qubits = len(self.graph.nodes())
        self.k=k
        if k != 2:
            self.num_qubits = k*len(self.graph) #TODO: implement logic for max values of k
        

    def build_circuit(self):
        """ 
        Convert graph to a cplex-problem of k-cut ( default k=2) and gets ising hamiltonian from it. Creates a circuit.
        updates self.: backend, circuit, cost_hamiltonian
        """
        conv = QuadraticProgramToQubo()
        is_k_cut = False
        if self.k != 2:
            is_k_cut = True
        self.solver = Solver(self.graph, relaxed = False, restrictions=self.restrictions, k=self.k) #use solver not to solve, but to get the qubo formulation - must not be relaxed!
        cost_hamiltonian = to_ising(conv.convert( self.solver.get_qp()))
        cost_hamiltonian_tuples = [(pauli, coeff) for pauli, coeff in zip([str(x) for x in cost_hamiltonian[0].paulis], cost_hamiltonian[0].coeffs)]
        self.build_backend()
        cost_hamiltonian = SparsePauliOp.from_list(cost_hamiltonian_tuples) #TODO: ensure these have the same coefficients - or these should be coefficients based on the 
                                                                            #weights of the connections in the graph, right? so its on the coeffs i should ensure weightedness?
        qc = None
        self.build_backend()
        if self.qaoa_variant =='vanilla':

            if self.warm_start:
                solver = Solver(self.graph, relaxed = True, restrictions=self.restrictions)
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
                qc = QAOAAnsatz(cost_operator = cost_hamiltonian, reps = params.depth, initial_state=initial_state, mixer_operator=mixer_state, flatten=True)

            else:
                qc = QAOAAnsatz(cost_operator = cost_hamiltonian, reps = params.depth)
            qc.measure_all()

        elif self.qaoa_variant =='multiangle': 
            multiangle_gammas = [[Parameter(f'γ_{l}_{i}') for i in range(len(self.graph.edges()))] for l in range(params.depth)]
            multiangle_betas = [[Parameter(f'β_{l}_{i}') for i in range(self.num_qubits)] for l in range(params.depth)]
    
            qc = QuantumCircuit(self.num_qubits)
            for _ in range(self.num_qubits): #initial state
                qc.h(_)

            for i in range(params.depth):
                for idx, edge in enumerate(self.graph.edge_list()):
                    qc.cx(edge[0], edge[1])
                    qc.rz(multiangle_gammas[i][idx], edge[1])
                    qc.cx(edge[0], edge[1])
                for idx in range(self.num_qubits):#TODO: add multiangle here
                    qc.rx(2*multiangle_betas[i][idx], idx)

            qc.measure_all()

        elif self.qaoa_variant =='recursive': #TODO: Fixx this
            if self.warm_start:
                solver = Solver(self.graph, relaxed = True, restrictions=self.restrictions)
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


            
            qaoa_mes = QAOA(sampler=BackendSampler(backend=self.backend), optimizer=COBYLA(), reps = params.depth, initial_point=[0.0,0.0], 
                            initial_state = initial_state, mixer = mixer_state)
            qaoa = MinimumEigenOptimizer(qaoa_mes) 
            self.rqaoa = RecursiveMinimumEigenOptimizer(qaoa, min_num_vars=7, min_num_vars_optimizer=exact) #TODO: Find exact¨
        #cost_operator = qc.cost_operator.to_operator()
        #print("Cost operator type: ", type(cost_operator))
        #mixer_operator = Operator(qc.mixer_operator)#why does this work on not warm start but not on warm start?
        #print("Cost operator: ", cost_operator)
        #print("Mixer operator: ", mixer_operator)
        #commutator = cost_operator @ mixer_operator - mixer_operator @ cost_operator
        #print("commutator: ", commutator) #TODO: decide on way to check these operators

        ##TODO: Scheck if circuit is flattened

        pm = generate_preset_pass_manager(optimization_level=3,backend=self.backend)
        candidate_circuit = pm.run(qc)
        self.circuit = candidate_circuit
        self.cost_hamiltonian = cost_hamiltonian

    def print_problem(self):
        if self.solver:
            print("problem:", self.solver.model.prettyprint())
        else:
            print('Solver is None. Run build_circuit or pass a solver (with a problem defined) in the constructor')


    def build_backend(self):

        if self.test:
            self.backend = GenericBackendV2(self.num_qubits) #FakeBrisbane is slow. For the test, where we onyl want to ensure stuff runs, we use a faster backend.
        elif self.simulation:
            self.backend = FakeBrisbane()
            print("You are running on the local simulator: ", self.backend.name)
        else:
            QiskitRuntimeService.save_account(channel="ibm_quantum", token=params.api_key, overwrite=True, set_as_default=True)
            service = QiskitRuntimeService(channel='ibm_quantum')
            self.backend = service.least_busy(min_num_qubits=127)
            print("You are running on the prestigious IBM machine ", self.backend)
        
    def draw_circuit(self):
        self.circuit.draw('mpl', fold=False, idle_wires=False)

    def get_init_params(self): 
        param_length = None #none so if its not changed its easier to see bugs - if it was 0 might be bugs further down the line
        if self.qaoa_variant == "vanilla":
            param_cost_length = 1
            param_mixer_length = 1

        elif self.qaoa_variant == "multiangle":
            param_cost_length = len(self.graph.edges())
            param_mixer_length = self.num_qubits


        match self.param_initialization: 
            case 'uniform':
                init_params = [np.random.uniform(0,np.pi,param_cost_length) + np.random.uniform(0,np.pi,param_mixer_length) for i in params.depth]
                return init_params
            case 'gaussian':
                init_params = [np.random.normal(0,np.pi,param_cost_length) + np.random.normal(0,np.pi,param_mixer_length) for i in params.depth]
                return init_params

            case 'machinelearning':
                raise NotImplementedError('Machine Learning not implemented yet. Use uniform or gaussian instead.') 



    def run(self):
        self.objective_func_vals = []
        init_params = self.get_init_params()

        start_time = time.time()
        if self.qaoa_variant == 'recursive':
            self.result = self.rqaoa.solve(self.graph.get_qp())

        else:
            with Session(backend = self.backend) as session:
                    estimator = Estimator(mode=session)
                    estimator.options.default_shots = 1000
                    if not self.test:
                            # Set simple error suppression/mitigation options
                            estimator.options.dynamical_decoupling.enable = True
                            estimator.options.dynamical_decoupling.sequence_type = "XY4"
                            estimator.options.twirling.enable_gates = True
                            estimator.options.twirling.num_randomizations = "auto"

                    result = minimize(
                    self.cost_func_estimator, 
                    init_params,
                    args= (self.circuit, self.cost_hamiltonian, estimator),
                    method = self.optimizer,
                    tol = 1e-2,
                    options={'disp': True}
                    )

        self.result = result
        print(self.result)
        self.circuit = self.circuit.assign_parameters(self.result.x)
        self.solution = self.calculate_solution()
        self.objective_value = self.evaluate_sample()
        

    def evaluate_sample(self) -> float:
        assert len(self.solution) == len(list(self.graph.nodes())), "The length of x must coincide with the number of nodes in the graph."
        return sum(self.solution[u] * (1 - self.solution[v]) + self.solution[v] * (1 - self.solution[u]) for u, v in set(self.graph.edge_list()))

    def cost_func_estimator(self,params, ansatz, hamiltonian, estimator):
        #TODO: see if this can be optimized
        #transform observable defined on virtual qubits to an observable defined on all physical qubits

        isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout) #TODO: What does this actually do?

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

        if not self.test:
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

        if not self.test:
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
        most_likely_bitstring = to_bitstring(most_likely,self.num_qubits)
        most_likely_bitstring.reverse()
        return most_likely_bitstring

    def compare_solutions(self, classic_solution):
        if not self.solution:
            raise ReferenceError("Solution not initalized yet. run()-function must be called to generate solution before it can be compared.")
        assert len(self.solution) == len(classic_solution[0]), 'Solutions not the same length.' #TODO: error relating to length of qubits for kcut which requires more qubits
        bools = [a==b for a,b in zip(classic_solution[0],self.solution)]
        bools_reversed =[a!=b for a,b in zip(classic_solution[0],self.solution)]
        print("Result quantum", self.solution, "Objective value: ", self.objective_value)
        print("Result input (classical)", classic_solution[0], "Objective Value: ", classic_solution[1])
        print("Same solution", all(bools) or all(bools_reversed)) #same cut but different partitions
        print("Same objective function value: ", classic_solution[1] == self.objective_value)

    def get_data_structures(self):

        pub = (self.circuit,)
        sampler = Sampler(mode=self.backend)
        sampler.options.default_shots=1000

        if not self.test:
        # Set simple error suppression/mitigation options
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.num_randomizations = "auto"

        job = sampler.run([pub], shots=int(1e4))
        return job

