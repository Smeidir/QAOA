import unittest
import rustworkx as rx
from qiskit.circuit.library import QAOAAnsatz
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from QAOA import QAOArunner  
from solver import Solver
from load_data import load_graph_from_csv
import params
import itertools
from MaxCutProblem import MaxCutProblem


class TestQAOArunner(unittest.TestCase):

    def setUp(self):
        """
        Set up a small test graph and initialize QAOArunner for testing.
        """
        # Create a simple graph (triangle graph for MaxCut)
        problem = MaxCutProblem()
        self.graph = problem.get_graph(6, create_random=True,random_weights=False)
        self.big_graph = load_graph_from_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', '11_nodes_links_scand.csv')))

        iterables = [params.supported_qaoa_variants, params.supported_param_inits]
        self.settings = []
        for t in itertools.product(*iterables):
            self.settings.append(t)
  
        # Initialize QAOArunner with simulation mode
        self.qaoa_runner = QAOArunner(graph=self.graph, simulation=True, test=True)
        self.solver = Solver(self.graph)  

        self.qaoa_runner.build_circuit()
        self.qaoa_runner.run()


    def test_build_circuit(self):
        """
        Test the build_circuit method.
        """

        circuit = self.qaoa_runner.circuit
        self.assertIsNotNone(circuit, "Circuit should not be None after building.")
        from qiskit import QuantumCircuit
        self.assertTrue(isinstance(circuit, QuantumCircuit), f"Circuit should be a QuantumCircuit. It is {type(circuit)}")

    def test_run(self):
        """
        Test the run method to ensure optimization works.
        """
        solution = self.qaoa_runner.solution
        self.assertIsNotNone(solution, "Solution should not be None after running optimization.")
        self.assertEqual(len(solution), self.qaoa_runner.num_qubits, "Solution length should match number of qubits.")

    def test_evaluate_sample(self):
        """
        Test the evaluation of the sample.
        """

        objective_value = self.qaoa_runner.evaluate_sample()
        self.assertTrue(objective_value is not None, "Objective value should not be None.")
    
    def test_draw_objective_value(self):
        """
        Test drawing the objective function evolution (visual validation needed).
        """
        # Visual check - ensure no exceptions are raised
        self.qaoa_runner.draw_objective_value()
    
    def test_solution_comparison(self):
        """
        Test comparing solutions against a known classical solution.
        """
        classical_solution = self.solver.solve()
        self.qaoa_runner.compare_solutions(classical_solution)

    def test_all_inits(self):


        for params in self.settings:
            qaoa = QAOArunner(self.graph, simulation=True, param_initialization=params[1],qaoa_variant=params[0])
            qaoa.build_circuit()
            qaoa.run()
    
        

if __name__ == '__main__':
    unittest.main()
