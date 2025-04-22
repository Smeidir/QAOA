import pytest
from qaoa.core.QAOA import QAOArunner
from qaoa.models.MaxCutProblem import MaxCutProblem

@pytest.mark.parametrize("variant", ["vanilla", "multiangle"])
@pytest.mark.parametrize("warm_start", [False, True])
@pytest.mark.parametrize("backend", ["statevector"])
@pytest.mark.parametrize("parameter_initialization", ["gaussian","static"])
def test_qaoa_run_completes(variant, warm_start, backend,parameter_initialization):
    graph = MaxCutProblem().get_erdos_renyi_graphs([5])[0]
    runner = QAOArunner(graph=graph, backend_mode=backend,
                        param_initialization=parameter_initialization,
                        qaoa_variant=variant,
                        warm_start=warm_start,
                        depth=1)
    runner.build_circuit()
    runner.run()
    assert runner.solution is not None
    assert isinstance(runner.objective_value, float)
    result_dict = runner.to_dict()
    assert all(value is not None for value in result_dict.values())