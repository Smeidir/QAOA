import time
import line_profiler
import matplotlib
from qiskit.quantum_info import Statevector, DensityMatrix,Operator, SparsePauliOp
import numpy as np
import rustworkx as rx
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import HGate, QAOAAnsatz
from qiskit.primitives import BackendSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator, StatevectorSimulator, Aer
from qiskit_aer.noise import NoiseModel
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, scipy_optimizer
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_optimization.algorithms import (MinimumEigenOptimizer,
                                            RecursiveMinimumEigenOptimizer)
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp, to_ising
from scipy.optimize import minimize



import params
from solver import Solver
from helper_functions import to_bitstring,to_bitstring_str
from MaxCutProblem import MaxCutProblem   #for testing purposes



class QAOArunner():
    """
    A class with all the functionality needed to create quantum circuit and run using the QAOA algorithm.
    inputs:
    Simulation: boolean, whether to run locally or on IBM cloud
    Graph: pygraph, the problem to solve
    initialization: string, the method of initializing the weights
    optimizer: what scipy optimizer to use.
    """
    def __init__(self, graph, backend_mode = 'statevector', param_initialization="gaussian",optimizer="COBYLA", qaoa_variant ='vanilla', 
                 warm_start=False,depth = 1, vertexcover = False,max_tol = 1e-8, amount_shots = 5000, lagrangian_multiplier = 2):
        
        if qaoa_variant not in params.supported_qaoa_variants:
            raise ValueError(f'Non-supported QAOA variant. Your param: {qaoa_variant} not in supported parameters:{params.supported_qaoa_variants}.')
        if param_initialization not in params.supported_param_inits:
            raise ValueError(f'Non-supported param initializer. Your param: {param_initialization} not in supported parameters:{params.supported_param_inits}.')
        if optimizer not in params.supported_optimizers:
            raise ValueError(f'Non-supported optimizer. Your param: {optimizer} not in supported parameters:{params.supported_optimizers}.')
        if backend_mode not in params.supported_backends:
            raise ValueError(f'Non-supported backend. Your param: {backend_mode} not in supported parameters:{params.supported_backends}.')
      
        self.graph = graph
        self.backend_mode = backend_mode
        self.param_initialization = param_initialization
        self.qaoa_variant = qaoa_variant
        self.optimizer = optimizer
        self.solution = None
        self.warm_start =warm_start
        self.objective_func_vals = []
        self.classical_objective_func_vals = []
        self.depth = depth
        self.vertexcover = vertexcover
        self.max_tol = max_tol
        self.amount_shots = amount_shots
        self.lagrangian_multiplier = lagrangian_multiplier
        self.solver = Solver(self.graph, lagrangian = self.lagrangian_multiplier, vertexcover = self.vertexcover) #use solver not to solve, but to get the qubo formulation - must not be relaxed!
        self.classical_solution,self.classical_objective_value = self.solver.solve()
        self.fev = 0 #0 quantum function evals, yet. Must initialize to increment.
        self.num_qubits = len(self.graph.nodes())

    def build_circuit(self):
        """ 
        Convert graph to a cplex-problem of k-cut ( default k=2) and gets ising hamiltonian from it. Creates a circuit.
        updates self.: backend, circuit, cost_hamiltonian
        """
        conv = QuadraticProgramToQubo()
        cost_hamiltonian = to_ising(conv.convert(self.solver.get_qp()))
        cost_hamiltonian_tuples = [(pauli, coeff) for pauli, coeff in zip([str(x) for x in cost_hamiltonian[0].paulis], cost_hamiltonian[0].coeffs)]
        self.build_backend()
        cost_hamiltonian = SparsePauliOp.from_list(cost_hamiltonian_tuples) 

        if not self.warm_start:

            if self.qaoa_variant =='vanilla':
                qc = QAOAAnsatz(cost_operator = cost_hamiltonian, reps = self.depth, flatten=True)

            elif self.qaoa_variant =='multiangle': 
                multiangle_gammas = [[Parameter(f'γ_{l}_{i}') for i in range(len(cost_hamiltonian_tuples))] for l in range(self.depth)]
                multiangle_betas = [[Parameter(f'β_{l}_{i}') for i in range(self.num_qubits)] for l in range(self.depth)]
                qc = QuantumCircuit(self.num_qubits)
                for _ in range(self.num_qubits): #initial state
                    qc.h(_)
                for i in range(self.depth):
                    for idx, (pauli, coeff) in enumerate(cost_hamiltonian_tuples):
                        z_placements = [i for i, interaction in enumerate(pauli) if interaction == 'Z']
                        if len(z_placements) == 1:
                            qc.rz(2*coeff*multiangle_gammas[i][idx], z_placements[0])
                        elif len(z_placements) ==2:
                            qc.cx(z_placements[0], z_placements[1])
                            qc.rz(2*multiangle_gammas[i][idx], z_placements[1])
                            qc.cx(z_placements[0], z_placements[1])
                        else:
                            raise ValueError('Amount of Z interactions in cost hamiltonian is over 2. You have', len(z_placements), 'interactions.')
                    for idx in range(self.num_qubits):
                        qc.rx(2*multiangle_betas[i][idx], idx)
        
            if self.qaoa_variant =='recursive': #TODO: Fixx this
                self.cum_fev = 0
                opti = None
                if self.optimizer == "COBYLA": opti = scipy_optimizer("COBYLA")
                qaoa_mes = QAOA(sampler=BackendSampler(backend=self.backend, options = {'shots': self.amount_shots}), 
                                optimizer=opti, reps = self.depth, initial_point=self.get_init_params())                          
                qaoa = MinimumEigenOptimizer(qaoa_mes) 
                self.rqaoa = RecursiveMinimumEigenOptimizer(qaoa, min_num_vars=3) #TODO: Find exact¨
     
        if self.warm_start:
            if self.qaoa_variant == 'vanilla':
                initial_state = QuantumCircuit(self.num_qubits)
                thetas = [-np.pi/2 + (1-2*x)* np.arctan(0.4) for x in self.classical_solution]

                for qubit in range(self.num_qubits): 
                    initial_state.ry(thetas[qubit],qubit)
                mixer_state = QuantumCircuit(self.num_qubits)
                mixer_param = Parameter('β')
                for qubit in range(self.num_qubits):
                    mixer_state.ry(thetas[qubit],qubit) 
                    mixer_state.rz(2*mixer_param, qubit)# Assign a placeholder beta parameter 
                    mixer_state.ry(-thetas[qubit],qubit)
                qc = QAOAAnsatz(cost_operator = cost_hamiltonian, reps = self.depth, initial_state=initial_state, mixer_operator=mixer_state, flatten=True)
            
            elif self.qaoa_variant == 'multiangle':
                thetas = [-np.pi/2 + (1-2*x)* np.arctan(0.4) for x in self.classical_solution]
                multiangle_gammas = [[Parameter(f'γ_{l}_{i}') for i in range(len(cost_hamiltonian_tuples))] for l in range(self.depth)]
                multiangle_betas = [[Parameter(f'β_{l}_{i}') for i in range(self.num_qubits)] for l in range(self.depth)]
                qc = QuantumCircuit(self.num_qubits)
                for qubit in range(self.num_qubits): 
                    qc.ry(thetas[qubit],qubit)

                for i in range(self.depth):
                    for idx, (pauli, coeff) in enumerate(cost_hamiltonian_tuples):
                        z_placements = [i for i, interaction in enumerate(pauli) if interaction == 'Z']
                        if len(z_placements) == 1:
                            qc.rz(2*coeff*multiangle_gammas[i][idx], z_placements[0])
                        elif len(z_placements) ==2:
                            qc.cx(z_placements[0], z_placements[1])
                            qc.rz(2*multiangle_gammas[i][idx], z_placements[1])
                            qc.cx(z_placements[0], z_placements[1])
                        else:
                            raise ValueError('Amount of Z interactions in cost hamiltonian is over 2. You have', len(z_placements), 'interactions.')
                    for idx in range(self.num_qubits):
                        qc.ry(thetas[idx],qubit) 
                        qc.rx(2*multiangle_betas[i][idx], idx)
                        qc.ry(-thetas[idx],qubit)

            
            if self.qaoa_variant =='recursive': #TODO: Fixx this
                raise ValueError('Recursive not implemented with warm start, due to inflexibility in qiskits recursive funcion.')
        qc.measure_all()
        commutation_tester = QAOAAnsatz(cost_operator = cost_hamiltonian, reps = self.depth) 
        cost_operator = commutation_tester.cost_operator.to_operator()
        mixer_operator = Operator(commutation_tester.mixer_operator)
        commutator = cost_operator @ mixer_operator - mixer_operator @ cost_operator
        if np.allclose(commutator.data, np.zeros((commutator.data.shape))):
            raise ArithmeticError("Operators commute.")


        ##TODO: Scheck if circuit is flattened

        if not self.qaoa_variant =='recursive':
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
        match self.backend_mode:
        
            case 'statevector':
                self.backend = StatevectorSimulator() 
                #print("You are running on the local StateVectorSimulator")
            case 'density_matrix_simulation':
                noise_model = NoiseModel.from_backend(FakeBrisbane())
                self.backend = AerSimulator(method='density_matrix',
                                noise_model=noise_model)
                #print("Running on: Density matrix simulator with noise")

            case 'noisy_sampling':
                self.backend = AerSimulator.from_backend(FakeBrisbane())
                
                #print("Running on: AerSimulator with noise")

            case 'quantum_backend':
                QiskitRuntimeService.save_account(channel="ibm_quantum", token=params.api_key, overwrite=True, set_as_default=True)
                service = QiskitRuntimeService(channel='ibm_quantum')
                self.backend = service.least_busy(min_num_qubits=127)
                print("Running on IBM quantum backend:", self.backend)
        
    def draw_circuit(self):
        fig = self.circuit.draw('mpl', fold=False, idle_wires=False)
        plt.tight_layout()
        plt.show()
        return fig

    def get_init_params(self): 
        param_cost_length = 1
        param_mixer_length = 1

        if self.qaoa_variant == "multiangle":

            param_cost_length = len(self.cost_hamiltonian)
            param_mixer_length = self.num_qubits

        match self.param_initialization: 
            case 'uniform':
                init_params = np.concatenate([
                    np.concatenate([np.random.uniform(0, 2*np.pi, param_cost_length), 
                                    np.random.uniform(0, np.pi, param_mixer_length)])
                    for _ in range(self.depth)
                ])
                init_params =init_params.flatten()
                return init_params
            case 'gaussian':

                init_params = np.concatenate([
                    np.concatenate([np.random.normal(np.pi,0.2,param_cost_length), 
                                    (np.random.normal(np.pi/2,0.1,param_mixer_length))])
                    for _ in range(self.depth)
                ])
                init_params =init_params.flatten()
                return init_params
            case 'static':
                optimized_params_dict = {1: [-np.pi/6, -np.pi/8],
                2: [-np.pi/6, -np.pi/8, -np.pi/4, -np.pi/13],
                3: [-np.pi*0.12641,np.pi*0.15244,np.pi*-0.24101,np.pi*-0.10299,np.pi*-0.27459,np.pi*-0.06517 ]} #https://arxiv.org/pdf/2102.06813
                if self.depth <= 3:
                    init_params = optimized_params_dict[self.depth]
                else:
                    step_size = 0.8/(self.depth-1)
                    final_values = 0.9/0.1
                    init_params = np.array([[0.1 + n*step_size, 0.9-n*step_size] for n in range(0,self.depth)])
                    init_params = init_params.flatten()

                if self.qaoa_variant == 'multiangle':
                    new_params = []
                    for i, param in enumerate(init_params):
                        if i%2 == 0: #cost parameter
                            new_params.extend([param]*param_cost_length)
                        else: #mixer parameter
                            new_params.extend([param]*param_mixer_length)
                    init_params = np.array(new_params)
                return init_params
                    
            case 'machinelearning':
                raise NotImplementedError('Machine Learning not implemented yet. Use uniform or gaussian instead.') 

    def callback_function(self, xk):
        print(f'Current solution: {xk} Current Objective value_{self.objective_func_vals[-1]}')
        

    def recursive_callback(self, *xk):
        if xk[0] == 1: #started new iteration
            self.cum_fev += self.fev #add amount last iteration  
        self.fev = xk[0]

    def run(self):
        self.objective_func_vals = []
        init_params = self.get_init_params()

        self.runtimes = []
        self.start_time = time.time()

        if self.qaoa_variant == 'recursive':
            result = self.rqaoa.solve(self.solver.get_qp())
            self.fev += self.cum_fev
            self.time_elapsed = time.time() -self.start_time
            self.result = result
            #self.circuit = self.rqaoa._optimizer hard to get the circuit out
            self.solution = result.x
            self.objective_value = result.fval
            return 

        
        match self.backend_mode:

            case 'statevector':
                clean_circuit = self.remove_measurements(self.circuit)
                result = minimize(
                self.cost_func_statevector, 
                init_params,
                args= (clean_circuit, self.cost_hamiltonian),
                method = self.optimizer,
                tol = self.max_tol,
                options={'disp': False, 'maxiter': 5000})


            case 'density_matrix_simulation':
                result = minimize(
                self.cost_func_density_matrix, 
                init_params,
                args= (self.circuit, self.cost_hamiltonian),
                method = self.optimizer,
                tol = self.max_tol,
                options={'disp': False, 'maxiter': 5000})
            
            case 'noisy_sampling'| 'quantum_backend' :
                estimator = Estimator(mode=self.backend)
                estimator.options.default_shots = self.amount_shots

                if self.backend_mode =='quantum_backend':
                    self.set_error_mitigation(estimator)
                isa_hamiltonian = self.cost_hamiltonian.apply_layout(self.circuit.layout) 
                result = minimize(
                self.cost_func_estimator, 
                init_params,
                args= (self.circuit, isa_hamiltonian, estimator),
                method = self.optimizer,
                tol = self.max_tol,
                options={'disp': False, 'maxiter': 5000})

        self.final_params = result.x
        self.time_elapsed = time.time() -self.start_time
        self.result = result
        self.fev = result.nfev
        #self.circuit = self.circuit.assign_parameters(self.result.x)
        self.solution = self.calculate_solution()
        self.objective_value = self.evaluate_solution()


    def run_no_optimizer(self, n = 50):
        self.objective_func_vals = []
        param_cost_length = 1
        param_mixer_length = 1

        if self.qaoa_variant == "multiangle":
            param_cost_length = len(self.graph.edges())
            param_mixer_length = self.num_qubits

        init_params = [
            np.concatenate([
                    np.concatenate([np.random.uniform(0, 2*np.pi, param_cost_length), 
                                    np.random.uniform(0, np.pi, param_mixer_length)])
                    for _ in range(self.depth)
                ]).flatten() for i in range(n)]
                
        self.runtimes = []
        self.start_time = time.time()
        if self.qaoa_variant == 'recursive':
                raise ValueError('Recursive not implemented for non-optimizer runs.')
        start_time = time.time()
        match self.backend_mode:
            case 'statevector':
        
                results = [self.cost_func_statevector(init_param, self.circuit, self.cost_hamiltonian) for init_param in init_params]

            case 'density_matrix_simulation':
                results = [self.cost_func_density_matrix(init_param, self.circuit, self.cost_hamiltonian) for init_param in init_params]
                #TODO: is this slow due to read/write erros?
            
            case 'noisy_sampling'| 'quantum_backend' :
                estimator = Estimator(mode=self.backend)
                estimator.options.default_shots = self.amount_shots
                if self.backend_mode == 'quantum_backend':
                    self.set_error_mitigation(estimator)
                start_time = time.time() #refresh time since estimator init can take time
                results = [self.cost_func_estimator(init_param, self.circuit, self.cost_hamiltonian, estimator=estimator) for init_param in init_params]

        best_result = np.min(results) #always min since the QAOA solves min energy
        best_index = results.index(best_result)
        best_parameters = init_params[best_index]

        self.final_params = best_parameters
        self.time_elapsed = time.time() -start_time
        self.result = best_result
        self.fev = n
        #self.circuit = self.circuit.assign_parameters(self.final_params)
        self.solution = self.calculate_solution()
        self.objective_value = self.evaluate_solution()

    def evaluate_solution(self) -> float:
        """Gives the objective value of self.solution. Must be used after run."""
        assert len(self.solution) == len(list(self.graph.nodes())), "The length of x must coincide with the number of nodes in the graph."
        solution_value = self.solver.evaluate_bitstring(self.solution)
        return solution_value
    
    def remove_measurements(self, circuit):
        """Return a new circuit with all measurements removed."""
        new_circuit = QuantumCircuit(circuit.num_qubits)
        for instruction in circuit.data:
            if instruction.operation.name != "measure":
                new_circuit.append(instruction.operation, instruction.qubits, instruction.clbits)
        return new_circuit


    def cost_func_estimator(self,params, ansatz, isa_hamiltonian, estimator):
        #TODO: see if this can be optimized
        #begin_time = time.time()
        pub = (ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])
        results = job.result()[0]
        cost = results.data.evs
        self.objective_func_vals.append(cost.item())
        #enclosing_scope_end_time = time.time()
        #self.runtimes.append(enclosing_scope_end_time-begin_time)
        return cost
    
    def cost_func_statevector(self, params, ansatz, hamiltonian):
        #begin_time = time.time()
        sv = Statevector.from_instruction(ansatz.assign_parameters(params))
        cost = np.real(sv.expectation_value(hamiltonian))
        self.objective_func_vals.append(cost)
        #enclosing_scope_end_time = time.time()
        #self.runtimes.append(enclosing_scope_end_time-begin_time)
        return cost
    
    def cost_func_density_matrix(self, params, ansatz, hamiltonian):
        circuit = self.remove_measurements(ansatz.assign_parameters(params))
        circuit.save_density_matrix()
        result = self.backend.run(circuit).result()
        rho = DensityMatrix(result.data(0)['density_matrix'])
        cost = np.real(rho.expectation_value(hamiltonian))
        self.objective_func_vals.append(cost)
        return cost
    
    def prob_best_solution(self,params):
        #TODO: see if this can be optimized
        
        final_distribution_int = self.get_bitstring_probabilities()

        keys = list(final_distribution_int.keys())
        values = np.array(list(final_distribution_int.values()))
        
        _,classical_value = self.solver.solve() 
        percent_chance_optimal = 0
        bitstrings = [list(reversed(to_bitstring(key, self.num_qubits))) for key in keys]
        evaluations = np.array([self.solver.evaluate_bitstring(bitstring) for bitstring in bitstrings], dtype = 'f')
        ideal_values = np.where(evaluations == classical_value, 1, 0)
        percent_chance_optimal = np.sum(values[evaluations == classical_value])
        return percent_chance_optimal


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

    def calculate_solution(self): #TODO: må da finnes en lettere måte?
        #TODO: support fior å finne flere av de mest sannsynlige?
        
        final_distribution_int = self.get_bitstring_probabilities()
        #print('final distribution int', final_distribution_int)
        #print(final_distribution_int)
        keys = list(final_distribution_int.keys())
        values = list(final_distribution_int.values())
        
        most_likely = keys[np.argmax(np.abs(values))]
        most_likely_bitstring = to_bitstring(most_likely,self.num_qubits)
        most_likely_bitstring.reverse()
        return most_likely_bitstring
    

    def get_bitstring_probabilities(self, params=None):
        """
        Returns a dictionary of bitstring probabilities from the current circuit.
        In sampling modes: returns normalized counts from sampling.
        In statevector/density matrix: returns exact probabilities.
        Optional to pass a set of parameters to test the qaoa on.
        """
  
        if (params is None) and (self.final_params is None): #truth value of array is ambigous
            raise ValueError('No parameters passed, and no final_params logged from an optimizer run. Please run the QAOA class or provide a parameter set.')
        params = params if params else self.final_params
        
        match self.backend_mode:
            case 'statevector':
                clean_circuit = self.remove_measurements(self.circuit)
                state = Statevector.from_instruction(clean_circuit.assign_parameters(params))
                probs = {int(k, 2): v for k, v in state.probabilities_dict().items()}
                return probs

            case 'density_matrix_simulation':
                clean_circuit = self.remove_measurements(self.circuit)
                clean_circuit.save_density_matrix()
                result = self.backend.run(clean_circuit.assign_parameters(params)).result()
                rho = DensityMatrix(result.data(0)['density_matrix'])
                # Projectors in computational basis
                probs = {}
                for i in range(2 ** self.num_qubits): #TODO: do this math by hand
                    number = i
                    projector = np.zeros((2 ** self.num_qubits, 2 ** self.num_qubits))
                    projector[i, i] = 1.0
                    probs[number] = np.real(np.trace(rho.data @ projector))
                return probs
            
            case 'noisy_sampling' | 'quantum_backend':
                bound_circuit = self.circuit.assign_parameters(params, inplace=False)
                pub = (bound_circuit)
                sampler = Sampler(mode=self.backend)
                sampler.options.default_shots = self.amount_shots
                if self.backend_mode == 'quantum_backend':
                    self.set_error_mitigation(sampler)
                job = sampler.run([pub])
                counts_int = job.result()[0].data.meas.get_int_counts()
                shots = sum(counts_int.values())
                final_distribution_int = {key: val/shots for key, val in counts_int.items()}
                return final_distribution_int
    

    def set_error_mitigation(self,backend):
        backend.options.dynamical_decoupling.enable = True
        backend.options.dynamical_decoupling.sequence_type = "XY4"
        backend.options.twirling.enable_gates = True
        backend.options.twirling.num_randomizations = "auto"

    def get_job_custom_circuit(self, circuit):
        pub = (circuit,self.final_params,)
        sampler = Sampler(mode=self.backend)
        sampler.options.default_shots=self.amount_shots

        if self.backend_mode == 'quantum_backend':
        # Set simple error suppression/mitigation options
            self.set_error_mitigation(sampler)

        job = sampler.run([pub])
        return job



    def get_prob_measure_optimal(self):

        final_distribution_int = self.get_bitstring_probabilities()
        keys = list(final_distribution_int.keys())
        values = list(final_distribution_int.values())

        percent_chance_optimal = 0
        _,valuess = self.solver.solve()
        
        for i in range(len(keys)):
            bitstring = list(reversed(to_bitstring(keys[i], self.num_qubits)))
            value = self.solver.evaluate_bitstring(bitstring)
            if value == valuess:
                #print('Bitstring', bitstring, 'has value', value, 'and probability ', values[i])
                percent_chance_optimal += values[i]
                
        return percent_chance_optimal
    
    def print_bitstrings(self):
        matplotlib.rcParams.update({"font.size": 10})
        final_distribution_int = self.get_bitstring_probabilities()

        final_bits = {to_bitstring_str(k,self.num_qubits):v for k, v in final_distribution_int.items()}
        values = np.abs(list(final_bits.values()))
        #top_4_values = sorted(values, reverse=True)[:4]
        positions = []
        for i,bitstr in enumerate(final_bits.keys()):
            bitstring = list(reversed([int(bit) for bit in bitstr]))
            if self.solver.evaluate_bitstring(bitstring) == self.classical_objective_value:
                positions.append(i)
           
        fig = plt.figure(figsize=(11, 6))
        ax = fig.add_subplot(1, 1, 1)
        plt.xticks(rotation=45)
        plt.title("Result Distribution")
        plt.xlabel("Bitstrings (reversed)")
        plt.ylabel("Probability")
        ax.bar(list(final_bits.keys()), list(final_bits.values()), color="tab:grey")
        for p in positions:
            ax.get_children()[int(p)].set_color("tab:purple")
        for i, bitstr in enumerate(final_bits.keys()):
            bitstring = list(reversed([int(bit) for bit in bitstr]))
            value = self.solver.evaluate_bitstring(bitstring, mark_infeasible=True)
            if isinstance(value, tuple):
                ax.text(i, final_bits[bitstr], f'{value[0]:.2f}', ha='center', va='bottom', color='red')
            else:
                ax.text(i, final_bits[bitstr], f'{value:.2f}', ha='center', va='bottom')
        plt.show()

if __name__ =='__main__':
    
    problem = MaxCutProblem()
    graphs = problem.get_erdos_renyi_graphs([5,7,9])

    graph = graphs[1]
    quantum = QAOArunner(graph=graph, 
                     backend_mode = 'statevector',
                     param_initialization= 'gaussian',
                     qaoa_variant='vanilla', 
                     optimizer='COBYLA',
                     warm_start=False,
                     depth = 1,
                     vertexcover=False,
                     amount_shots = 1000,
                     max_tol = 1e-2,
                     lagrangian_multiplier=2
                     )

    quantum.build_circuit()
    quantum.run()
    quantum = QAOArunner(graph=graph, 
                     backend_mode = 'statevector',
                     param_initialization= 'gaussian',
                     qaoa_variant='multiangle', 
                     optimizer='COBYLA',
                     warm_start=False,
                     depth = 1,
                     vertexcover=False,
                     amount_shots = 1000,
                     max_tol = 1e-2,
                     lagrangian_multiplier=2
                     )

    quantum.build_circuit()
    quantum.run()
    quantum = QAOArunner(graph=graph, 
                     backend_mode = 'statevector',
                     param_initialization= 'gaussian',
                     qaoa_variant='vanilla', 
                     optimizer='COBYLA',
                     warm_start=True,
                     depth = 1,
                     vertexcover=False,
                     amount_shots = 1000,
                     max_tol = 1e-2,
                     lagrangian_multiplier=2
                     )

    quantum.build_circuit()
    quantum.run()
    quantum = QAOArunner(graph=graph, 
                     backend_mode = 'statevector',
                     param_initialization= 'gaussian',
                     qaoa_variant='multiangle', 
                     optimizer='COBYLA',
                     warm_start=True,
                     depth = 1,
                     vertexcover=False,
                     amount_shots = 1000,
                     max_tol = 1e-2,
                     lagrangian_multiplier=2
                     )

    quantum.build_circuit()
    quantum.run()

