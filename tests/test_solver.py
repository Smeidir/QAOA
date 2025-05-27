import pytest
from qaoa.models.solver import create_solver
from qaoa.models.MaxCutProblem import MaxCutProblem

@pytest.mark.parametrize("problem_type", ["maxcut", "minvertexcover"])
def test_solver_output(problem_type):
    graph = MaxCutProblem().get_erdos_renyi_graphs([5])[0]
    solver = create_solver(graph,  problem_type)
    solution, value = solver.solve()
    assert isinstance(solution, list)
    assert isinstance(value, float)