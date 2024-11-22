import unittest
import rustworkx as rx
from qiskit.circuit.library import QAOAAnsatz
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from QAOA import QAOArunner  
from load_data import load_graph_from_csv
import params

class TestQAOArunner(unittest.TestCase):

    def setUp(self):
        """
        Set up a small test graph and initialize QAOArunner for testing.
        """
        # Create a simple graph (triangle graph for MaxCut)
        self.graph = rx.PyGraph()
        self.graph.add_nodes_from([0, 1, 2])
        self.graph.add_edges_from([(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])  # Edges with weights
        self.big_graph = load_graph_from_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', '11_nodes_links_scand.csv')))
        all_combos = zip(params.supported_optimizers,params.supported_param_inits, params.supported_optimizers) # deosnt work - fix
        
        
        
        
        # Initialize QAOArunner with simulation mode
        self.qaoa_runner = QAOArunner(graph=self.graph, simulation=True, warm_start=True)

    def test_initialization(self):
        """
        Test the initialization of QAOArunner.
        """
        self.assertEqual(len(self.qaoa_runner.graph.nodes()), 3)
        self.assertTrue(self.qaoa_runner.simulation)
        self.assertEqual(self.qaoa_runner.num_qubits, 3)

    def test_build_circuit(self):
        """
        Test the build_circuit method.
        """
        self.qaoa_runner.build_circuit()
        circuit = self.qaoa_runner.circuit
        self.assertIsNotNone(circuit, "Circuit should not be None after building.")
        self.assertTrue(isinstance(circuit, QAOAAnsatz), f"Circuit should be a QAOAAnsatz. It is {type(circuit)}")

    def test_run(self):
        """
        Test the run method to ensure optimization works.
        """
        self.qaoa_runner.build_circuit()
        self.qaoa_runner.run()
        solution = self.qaoa_runner.solution
        self.assertIsNotNone(solution, "Solution should not be None after running optimization.")
        self.assertEqual(len(solution), self.qaoa_runner.num_qubits, "Solution length should match number of qubits.")

    def test_evaluate_sample(self):
        """
        Test the evaluation of the sample.
        """
        self.qaoa_runner.build_circuit()
        self.qaoa_runner.run()
        objective_value = self.qaoa_runner.evaluate_sample()
        self.assertTrue(objective_value is not None, "Objective value should not be None.")
    
    def test_draw_objective_value(self):
        """
        Test drawing the objective function evolution (visual validation needed).
        """
        self.qaoa_runner.build_circuit()
        self.qaoa_runner.run()
        # Visual check - ensure no exceptions are raised
        self.qaoa_runner.draw_objective_value()
    
    def test_solution_comparison(self):
        """
        Test comparing solutions against a known classical solution.
        """
        classical_solution = ([0, 1, 0], 3.0)  # Example classical solution
        self.qaoa_runner.build_circuit()
        self.qaoa_runner.run()
        self.qaoa_runner.compare_solutions(classical_solution)
    
    def test_all_params(self):

        all_combos = zip(params.supported_optimizers,params.supported_param_inits)
        print(all_combos)
        

if __name__ == '__main__':
    unittest.main()
