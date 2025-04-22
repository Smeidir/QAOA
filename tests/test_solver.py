from qaoa.models.solver import Solver
from qaoa.models.MaxCutProblem import MaxCutProblem

def test_solver_output():
    graph = MaxCutProblem().get_erdos_renyi_graphs([5])[0]
    solver = Solver(graph)
    solution, value = solver.solve()
    assert isinstance(solution, list)
    assert isinstance(value, float)