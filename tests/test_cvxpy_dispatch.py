"""Optional prepared dispatch preserves checked engineering semantics."""

from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import pytest

from operange import _cvxpy_backend as backend
from operange._linear_recourse import compile_system
from examples.linear_dispatch import example
from tests.test_linear_diagnosis import evidence
from tests.test_linear_process import cases, portable, small_model


def test_missing_backend_is_explicit_and_default_still_works(monkeypatch):
    def unavailable(name):
        raise ImportError("Injected missing optional dependency")

    monkeypatch.setattr(backend, "import_module", unavailable)
    model, domain = example()
    claim = model.as_claim(domain)
    result = claim.evaluate_result({"dryer": 10, "evaporator": 6}, backend="cvxpy")
    assert result.execution == "unsupported"  # nosec B101
    assert "operange[cvxpy]" in result.to_json()  # nosec B101
    assert claim.audit_result(backend="cvxpy").execution == "unsupported"  # nosec B101
    assert claim.audit_result().payload.verdict == "fail"  # nosec B101
    assert claim.audit_result(backend="misspelled").execution == "invalid"  # nosec B101
    portable(result)


@pytest.fixture
def cvxpy():
    cp = pytest.importorskip("cvxpy")
    backend.clear_cache()
    yield cp
    backend.clear_cache()


def test_dispatch_audit_conflict_and_relief_parity(cvxpy):
    model, domain = example()
    claim = model.as_claim(domain)
    baseline, result = claim.audit_result(), claim.audit_result(backend="cvxpy")
    assert result.contract == baseline.contract  # nosec B101
    assert result.request["backend"] == "cvxpy"  # nosec B101
    assert result.payload.verdict == baseline.payload.verdict == "fail"  # nosec B101
    assert result.payload.coverage.method == "complete_finite"  # nosec B101
    assert [e.payload.feasibility for e in result.supporting_evaluations] == [
        e.payload.feasibility for e in baseline.supporting_evaluations
    ]  # nosec B101
    portable(result)
    diagnosis = claim.evaluate_result(
        {"dryer": 12, "evaporator": 8},
        backend="cvxpy",
        diagnose=True,
        relief={"constraint": "shared_fuel", "maximum": 2},
    )
    conflict, relief = evidence(diagnosis, "conflict"), evidence(diagnosis, "relief")
    assert diagnosis.payload.feasibility == "infeasible"  # nosec B101
    assert conflict["irreducible"]  # nosec B101
    assert set(conflict["constraint_refs"]) == {"meet_load", "shared_fuel"}  # nosec B101
    optimum = Fraction(1) - 3 * Fraction(1e-8)
    assert Fraction(relief["lower"]) <= optimum <= Fraction(relief["upper"])  # nosec B101
    assert relief["resolution"] == "minimum_verified"  # nosec B101
    portable(diagnosis)
    unreachable = claim.evaluate_result(
        {"dryer": 12, "evaporator": 8},
        backend="cvxpy",
        relief={"constraint": "shared_fuel", "maximum": 0.5},
    )
    assert evidence(unreachable, "relief")["resolution"] == "unreachable"  # nosec B101


def test_cache_reuses_compilation_but_updates_loads_and_objectives(cvxpy, monkeypatch):
    builds = []
    original = backend.PreparedLinearProgram.__init__

    def record(self, *a, **kw):
        builds.append(kw["phase_one"])
        original(self, *a, **kw)

    monkeypatch.setattr(backend.PreparedLinearProgram, "__init__", record)
    model = small_model()
    claim = model.as_claim(cases(model, 0.25, 0.75))
    first = claim.evaluate_result({"load": 0.25}, backend="cvxpy")
    second = claim.evaluate_result({"load": 0.75}, backend="cvxpy")
    assert first.payload.feasibility == second.payload.feasibility == "feasible"  # nosec B101
    assert builds == [True]  # nosec B101
    assert (
        first.result_id
        == claim.evaluate_result({"load": 0.25}, backend="cvxpy").result_id
    )  # nosec B101
    system = compile_system(claim, {"load": 0.25})
    low = backend.solve_bounded(system, [1], 1e-9)
    high = backend.solve_bounded(system, [-1], 1e-9)
    assert low.point == pytest.approx([0.25]) and high.point == pytest.approx([1])  # nosec B101
    assert builds == [True, False]  # nosec B101
    assert system.objective_bound([1], low.multipliers)[0] == Fraction(1, 4)  # nosec B101


def test_mutable_workspaces_are_thread_local_and_bounded():
    backend.clear_cache()
    program = backend.cached_program("test", object)
    with ThreadPoolExecutor(max_workers=1) as executor:
        other = executor.submit(backend.cached_program, "test", object).result()
    assert other is not program  # nosec B101
    for i in range(backend._CACHE_SIZE):
        backend.cached_program(i, object)
    assert backend.cached_program("test", object) is not program  # nosec B101
    backend.clear_cache()


@pytest.mark.parametrize("fault", ("error", "infeasible", "nan", "wrong_dispatch"))
def test_solver_failure_or_bad_candidates_never_prove_infeasibility(
    cvxpy, monkeypatch, fault
):
    model = small_model(scale=1e12 if fault == "wrong_dispatch" else 1)
    claim = model.as_claim(cases(model, 0.5))
    # Ensure stale values from a previous solve cannot be consumed on failure.
    if fault != "wrong_dispatch":
        assert (
            claim.evaluate_result({"load": 0.5}, backend="cvxpy").payload.feasibility
            == "feasible"
        )  # nosec B101

    if fault in ("error", "infeasible"):

        def broken(problem, **options):
            if fault == "error":
                raise cvxpy.error.SolverError("Injected failure")
            problem._status = cvxpy.INFEASIBLE

        monkeypatch.setattr(cvxpy.Problem, "solve", broken)
    else:
        monkeypatch.setattr(
            backend,
            "solve_bounded",
            lambda *a, **kw: backend.CandidateSolve(
                [float("nan") if fault == "nan" else 0], [0], []
            ),
        )
    result = claim.evaluate_result({"load": 0.5}, backend="cvxpy")
    assert result.execution == "unresolved"  # nosec B101
    assert result.payload.feasibility == "unknown"  # nosec B101
    assert not any(e.subject == "recourse_infeasibility" for e in result.evidence)  # nosec B101
    portable(result)
