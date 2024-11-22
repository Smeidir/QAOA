import cplex
from docplex.mp.model import Model
from matplotlib import pyplot as plt
from rustworkx import NoEdgeBetweenNodes
import rustworkx as rx
from rustworkx import is_connected
from rustworkx.visualization import mpl_draw as draw_graph
from load_data import load_graph_from_csv
from mystic.solvers import fmin, fmin_powell
from mystic.monitors import VerboseMonitor
from qiskit_ibm_runtime import QiskitRuntimeService
import params
import random
import numpy as np
from MaxCutProblem import MaxCutProblem
import time
from qiskit_optimization.translators import from_docplex_mp, to_ising
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit.primitives import Sampler, BackendSamplerV2, BackendSampler #samplre is deprecated, but need it to run. Why?
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizer,
    SolutionSample,
    OptimizationResultStatus,
)
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_optimization.algorithms import WarmStartQAOAOptimizer
from qiskit_optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

class Solver():
    """
    Class which contains the ordinary cplex solver.
    Used for getting solutions to compare with quantum solution.
    TODO: Add support for max k-cut
    """
    
    def __init__(self, graph, relaxed = False):

        """
        Initializes the model with the given problem, but does not solve.
        If relaxed is set to true it changes the behaviour when solve is called. it will then find a local optima, 
        as cplex cannot handle non-convex continous variables.
        """
        self.graph = graph
        self.model = Model(name="MaxCut")
        self.relaxed = relaxed

        if relaxed: 
            self.variables = self.model.continuous_var_list(len(self.graph),lb=0,ub=1, name='x')
            self.model.parameters.optimalitytarget =2 #local minima
        else:
            self.variables = self.model.binary_var_list(len(self.graph), name='x')
        
        objective = 0

        for i,var in enumerate(self.variables): #hvorfor går jeg ikke bare gjennom edges??
            for j,var2 in enumerate(self.variables):
                if i != j and i> j:  
                    try:
                        graph.get_edge_data(i,j)
                        objective += var + var2 - 2*var*var2
                    except NoEdgeBetweenNodes:
                        pass
        self.objective = objective
        
        # Add equality constraint: sum of variables equals half the number of nodes
        #self.model.add_constraint(self.model.sum(self.variables) >= len(self.graph) // 2)
        # Add constraint: sum of x0, x1, and x2 must be over 2
        self.model.add_constraint(self.variables[0] + self.variables[1] + self.variables[2] == 1)
        self.model.add_constraint(self.variables[3] + self.variables[4] + self.variables[5] == 1)
        self.model.add_constraint(self.variables[6] + self.variables[7] + self.variables[8] == 1)
        self.model.add_constraint(self.variables[9] + self.variables[10]  == 1)

        self.model.objective=objective
        self.model.maximize(self.objective)

    def get_qp(self):
        return from_docplex_mp(self.model)
        

    def solve(self, verbose = False):
        """
        Solves the problem based on parameter relaxed.
        Returns bitstring, solution_value
        """
        
        if verbose:
            print(f'Objective to maximize: {self.objective} for relaxed = {self.relaxed}')
        self.model.maximize(self.objective)

        solution = self.model.solve()
        bitstring = [var.solution_value for var in self.variables]
        if verbose:
            print(solution.get_objective_value(), bitstring)
        return bitstring, solution.get_objective_value()

    
    def plot_result(self):
        """
        Plots graph of partition. Must be run after solve.
        """
        bitstring = [int(var.solution_value) for var in self.variables]


        colors = ["tab:grey" if i == 0 else "tab:purple" for i in bitstring]
        pos, default_axes = rx.spring_layout(self.graph), plt.axes(frameon=True)
        rx.visualization.mpl_draw(self.graph, node_color=colors, node_size=100, alpha=0.8, pos=pos)

#mystic is legacy for testing - unless we need it later.

def max_cut_objective(x, graph):
    objective = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if graph.has_edge(i, j):
                objective += x[i] + x[j] - 2 * x[i] * x[j]
    return -objective  # Minimize the negative to maximize the original objective
def solve_with_mystic(graph):
    x0 = [0.0] * len(graph)  # Initial guess
    bounds = [(0,1) for _ in range(len(graph))]
    
    #monitor = VerboseMonitor(100)

    result = fmin_powell(max_cut_objective, x0, args=(graph,), bounds=bounds,maxiter=1000, ftol=1e-6, verbose=False)
    
    solution_values = result
    objective_value = -max_cut_objective(solution_values, graph)
    return solution_values, objective_value

#bitstring, objective_value = solve_with_mystic(load_graph_from_csv('data/11_nodes_links_scand.csv'))
#print(f'Objective value: {objective_value:.6f}, Bitstring: {[f"{bit:.4f}" for bit in bitstring]}')
def is_almost_integer_solution(solution, tolerance=0.1):
    
    conclusion = all(abs(x - round(x)) < tolerance for x in solution)
    #if not conclusion:
     #   print('Not integers:' ,solution)
    return conclusion

num_graphs = 100
sizes = range(10, 51)
mystic_better_count = 0
integer_solution_count = 0
results = []

problem = MaxCutProblem()

def format_qaoa_samples(samples, max_len: int = 10):
    qaoa_res = []
    for s in samples:
            qaoa_res.append(("".join([str(int(_)) for _ in s.x]), s.fval, s.probability))

    res = sorted(qaoa_res, key=lambda x: -x[1])[0:max_len]

    return [(_[0] + f": value: {_[1]:.3f}, probability: {1e2*_[2]:.1f}%") for _ in res]


if __name__ == "__main__":
    solver = Solver(load_graph_from_csv('data/11_nodes_links_scand.csv'), True)
    solver.solve(True)


    backend = GenericBackendV2(num_qubits=11)
    qaoa_mes = QAOA(sampler=BackendSampler(backend=backend), optimizer=COBYLA(), initial_point=[0.0,0.0])
    exact_mes = NumPyMinimumEigensolver()
    exact = MinimumEigenOptimizer(exact_mes) 


    graph = Solver(load_graph_from_csv('data/11_nodes_links_scand.csv'))

    print('Cplex solver:', graph.solve()[1])

    print(graph.get_qp().prettyprint())
    conv = QuadraticProgramToQubo()

    print("QUBO:", conv.convert(graph.get_qp()))
    print("ising:", to_ising(conv.convert(graph.get_qp())))
    ising = to_ising(conv.convert(graph.get_qp()))

    print("ising" , ising)
    print("len ", [str(x) for x in ising[0].paulis])


    exact_result = exact.solve(conv.convert(graph.get_qp()))

    qaoa = MinimumEigenOptimizer(qaoa_mes) 
    #print("Exact:",exact_result.prettyprint())

    solution = qaoa.solve(graph.get_qp())
    print("QAOA:", format_qaoa_samples(solution.samples))

    #QiskitRuntimeService.save_account(channel="ibm_quantum", token=params.api_key, overwrite=True, set_as_default=True)
    #service = QiskitRuntimeService(channel='ibm_quantum')
    #backend = service.least_busy(min_num_qubits=127)
    #print(backend)
    qaoa_mes = QAOA(sampler=BackendSampler(backend=backend), optimizer=COBYLA(), initial_point=[0.0,0.0])
    qaoa = MinimumEigenOptimizer(qaoa_mes) 

    rqaoa = RecursiveMinimumEigenOptimizer(qaoa, min_num_vars=7, min_num_vars_optimizer=exact)
    rqaoa_result = rqaoa.solve(graph.get_qp())
    print("RQAOA:" ,rqaoa_result.prettyprint())

"""

pres = CplexOptimizer()
pres.parameters.optimalitytarget = 2
ws_qaoa = WarmStartQAOAOptimizer(
    pre_solver=pres, relax_for_pre_solver=True, qaoa=qaoa_mes, epsilon=0.0
)
wsqaoa_result = ws_qaoa.solve(graph.get_qp())
print("WSQAOA:" ,wsqaoa_result.prettyprint())"""

"""


for _ in range(num_graphs):
    print("1 done")
    size = random.choice(sizes)
    graph = problem.get_graph(size, create_random=True)
    
    solver = Solver(graph, relaxed=False)
    cplex_bitstring, cplex_objective = solver.solve()
    
    mystic_bitstring, mystic_objective = solve_with_mystic(graph)
    
    if mystic_objective > cplex_objective:
        mystic_better_count += 1
    
    if is_almost_integer_solution(mystic_bitstring):
        integer_solution_count += 1
    
    results.append((size, cplex_objective, mystic_objective))

# Plot results
sizes, cplex_results, mystic_results = zip(*results)
plt.figure(figsize=(12, 6))
plt.plot(sizes, cplex_results, 'o', label='CPLEX')
plt.plot(sizes, mystic_results, 'x', label='Mystic')
plt.xlabel('Graph Size')
plt.ylabel('Objective Value')
plt.legend()
plt.title('Comparison of CPLEX and Mystic Solutions')
plt.show()

print(f'Mystic had a better solution {mystic_better_count} times out of {num_graphs}')
print(f'Mystic solution was only integers {integer_solution_count} times out of {num_graphs}')

""""""
sizes = range(10, 181, 10)
runtimes = []
plot_sizes = [number for number in sizes for i in range(10)]


for size in sizes:
    runtimes2 = []
    for i in range(1):
        graph = problem.get_graph(size, create_random=True, random_weights=True)
        solver = Solver(graph, relaxed=False)
        
        start_time = time.time()
        solver.solve(verbose=False)
        solution = solver.solve()
        end_time = time.time()
    
        runtime = end_time - start_time
        runtimes2.append(runtime)
        #print(f'Size: {size}, Runtime: {runtime:.6f} seconds')
    runtimes.append(np.mean(runtimes2))
    print("Done with ", size)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(sizes, runtimes, 'o-', label='CPLEX Runtime')
plt.xlabel('Graph Size')
plt.ylabel('Runtime (seconds)')
plt.legend()
plt.title('CPLEX Solver Runtime vs Graph Size')
plt.show()"""