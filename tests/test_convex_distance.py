"""Distance certificates use exact norm/support bounds and physical replay."""

from dataclasses import replace
from fractions import Fraction as F
from math import sqrt

import pytest

from operange import (
    BoxSet,
    BudgetSet,
    FrozenController,
    Intersection,
    NormalizedL2,
    NormalizedLInf,
    ParameterSpace,
    Product,
    Union,
)
from operange import _convex_distance as backend
from operange import _cvxpy_backend as cvxpy_backend
from examples.distance_and_relief import distance_example
from examples.frozen_controllers import envelope_example
from examples.uncertainty_compositions import convex_examples
from tests.test_affine_distance import (
    candidate,
    claim_for,
    scalar_claim,
    verify_minimum,
    portable,
)
from tests.test_cvxpy_intersections import circle


@pytest.fixture
def cvxpy():
    cp = pytest.importorskip("cvxpy")
    cvxpy_backend.clear_cache()
    yield cp
    cvxpy_backend.clear_cache()


def branch(result, index=0):
    return next(
        e.details for e in result.evidence if e.evidence_id == f"branch:{index}"
    )


def test_euclidean_metric_keeps_scales_units_and_outside_origins_explicit():
    claim = claim_for()
    space = claim.domain.space
    metric = NormalizedL2(space)
    assert metric.measure({"dryer": 12, "evaporator": 8}) >= sqrt(2)  # nosec B101
    assert (
        metric.measure({"dryer": 12, "evaporator": 8}, {"dryer": 6, "evaporator": 0})
        == 5
    )  # nosec B101
    assert metric.to_manifest()["metric"] == "normalized_l2"  # nosec B101
    tiny = float.fromhex("0x0.0000000000001p-1022")
    scalar = replace(space.coordinates[0], nominal=0, scale=2)
    assert NormalizedL2(ParameterSpace((scalar,))).measure({"dryer": tiny}) == tiny  # nosec B101
    with pytest.raises(ValueError):
        metric.measure({"dryer": float("inf"), "evaporator": 8})


@pytest.mark.parametrize(
    "metric, expected", ((NormalizedLInf, 0.75), (NormalizedL2, sqrt(2) * 0.75))
)
def test_box_distances_match_known_optima_and_proof_recomputes(cvxpy, metric, expected):
    claim = claim_for(requirements=("steam_header",))
    claim = replace(claim, distance=metric(claim.domain.space))
    result = claim.boundary_result(backend="cvxpy")
    verify_minimum(result, expected)
    data = branch(result)
    proof = data["lower_certificate"]
    v = list(map(F, proof["norm_dual_exact"]))
    assert (sum(x * x for x in v) if metric is NormalizedL2 else sum(map(abs, v))) <= 1  # nosec B101
    multiplier = F(proof["target_multiplier_exact"])
    assert multiplier >= 0  # nosec B101
    w = {
        n: multiplier * a - m * x
        for n, a, m, x in zip(
            data["coordinates"],
            map(F, data["target_coefficients_exact"]),
            map(F, data["distance_scales_exact"]),
            v,
        )
    }
    assert w == {n: F(a) for n, a in proof["support_direction_exact"].items()}  # nosec B101
    support = proof["support_certificate"]
    residual, total = dict(w), F(0)
    for factor in support["factors"]:
        local = factor["support"]
        total += F(local["upper"])
        for n, alpha, beta in zip(
            factor["coordinates"], factor["alpha_exact"], factor["beta_exact"]
        ):
            a = F(local["coefficients"][n])
            residual[n] -= a * F(alpha)
            total -= a * F(beta)
    total += sum(
        max(a * F(b) for b in data["enclosure_exact"][n]) for n, a in residual.items()
    )
    assert total == F(proof["support_upper_exact"])  # nosec B101
    lower = (
        multiplier * F(data["target_rhs_exact"])
        + sum(x * F(d) for x, d in zip(v, data["distance_offsets_exact"]))
        - total
    )
    assert lower == F(proof["lower_exact"])  # nosec B101
    if metric is NormalizedL2:
        point = candidate(result).request["realization"]
        squared = sum(
            ((F(point[c.name]) - F(c.nominal)) / F(c.scale)) ** 2
            for c in claim.distance.space.coordinates
        )
        assert (
            squared
            == F(data["distance_squared_exact"])
            <= F(result.payload.search.upper.value) ** 2
        )  # nosec B101


@pytest.mark.parametrize(
    "metric, expected",
    ((NormalizedLInf, 0.855), (NormalizedL2, sqrt(0.5**2 + 0.855**2))),
)
def test_clipped_ellipsoid_positive_threshold_replays(cvxpy, metric, expected):
    model, domain = distance_example()
    claim = model.as_claim(domain, distance=metric(domain.space))
    result = claim.breaking_result(
        backend="cvxpy", violation_margins={"header_capacity": 0.01}
    )
    verify_minimum(result, expected)
    assert result.payload.verdict == "fail"  # nosec B101
    point = result.payload.witness.realizations[0]
    assert backend.exact_member(backend.leaves(domain), point)  # nosec B101
    assert any(
        c.assessment == "violated"
        for c in claim.evaluate_result(point).payload.constraint_checks
    )  # nosec B101


def test_domain_constraints_and_distance_coordinate_orders_are_independent(cvxpy):
    claim = claim_for(requirements=("steam_header",))
    envelope = claim.domain
    domain = Product(tuple(BoxSet((p,)) for p in reversed(envelope.parameters)))
    distance = NormalizedL2(
        ParameterSpace(
            tuple(
                replace(c, nominal=c.nominal - 3, scale=1)
                for c in envelope.space.coordinates
            )
        )
    )
    result = replace(claim, domain=domain, distance=distance).boundary_result(
        backend="cvxpy"
    )
    verify_minimum(result, sqrt(2) * 4.5)


def test_simplex_specification_and_budget_targets(cvxpy):
    model, domain = convex_examples()["specified_feed"]
    model = replace(model, requirements=(replace(model.requirements[0], limit=0.7),))
    result = model.as_claim(
        domain, distance=NormalizedL2(domain.space)
    ).boundary_result(backend="cvxpy")
    verify_minimum(result, sqrt(2) * 0.2)
    assert sum(map(F, candidate(result).request["realization"].values())) == 1  # nosec B101
    claim = claim_for()
    restricted = Intersection(
        (claim.domain, BudgetSet(claim.domain, 1)), backend="cvxpy"
    )
    unreachable = replace(claim, domain=restricted).boundary_result(backend="cvxpy")
    assert unreachable.payload.search.resolution == "unreachable"  # nosec B101
    for i in range(2):
        proof = branch(unreachable, i)["infeasibility_certificate"]
        assert F(proof["upper_exact"]) < F(proof["rhs_exact"])  # nosec B101
    portable(unreachable)


def test_frozen_controller_euclidean_distance_and_rounding_replay(cvxpy):
    claim = envelope_example()
    domain = Intersection((claim.domain, circle(claim.domain)), backend="cvxpy")
    claim = replace(claim, domain=domain, distance=NormalizedL2(domain.space))
    loaded = FrozenController.from_json(claim.freeze().to_json()).as_claim()
    assert type(loaded.distance) is NormalizedL2  # nosec B101
    assert loaded.contract.ref == claim.contract.ref  # nosec B101
    # The fuel limit is unreachable on this smaller envelope; the exact support
    # proof also includes the saved controller's command rounding allowance.
    result = loaded.boundary_result(backend="cvxpy", constraints=("shared_fuel",))
    assert result.payload.search.resolution == "unreachable"  # nosec B101
    portable(result)
    full = envelope_example()
    full = replace(full, distance=NormalizedL2(full.domain.space))
    loaded = FrozenController.from_json(full.freeze().to_json()).as_claim()
    failure = loaded.breaking_result(
        backend="cvxpy",
        constraints=("shared_fuel",),
        violation_margins={"shared_fuel": 0.01},
    )
    assert failure.payload.search.resolution == "minimum_verified"  # nosec B101
    assert failure.payload.witness.kind == "fixed_policy_failure"  # nosec B101
    replay = loaded.evaluate_result(failure.payload.witness.realizations[0])
    assert replay.payload.feasibility == "infeasible"  # nosec B101
    portable(failure)


def test_conic_solver_exception_preserves_only_independent_evidence(cvxpy, monkeypatch):
    def broken(*a, **k):
        raise cvxpy.error.SolverError("Injected solver failure")

    monkeypatch.setattr(cvxpy.Problem, "solve", broken)
    result = claim_for().boundary_result(backend="cvxpy")
    assert result.payload.search.resolution == "unresolved"  # nosec B101
    assert result.payload.search.upper.state == "unknown"  # nosec B101
    assert result.payload.search.lower.value == 0  # nosec B101
    portable(result)


def test_repeated_thresholds_reuse_prepared_geometry(cvxpy, monkeypatch):
    original, programs = backend.PreparedDistance.__init__, []

    def counted(self, problem):
        original(self, problem)
        programs.append(self)

    monkeypatch.setattr(backend.PreparedDistance, "__init__", counted)
    claim = claim_for(requirements=("steam_header",))
    claim.boundary_result(backend="cvxpy")
    claim.breaking_result(backend="cvxpy", violation_margins={"steam_header": 0.01})
    assert len(programs) == 1 and programs[0].problem.is_dcp(dpp=True)  # nosec B101


@pytest.mark.parametrize(
    "fault",
    ("solver_error", "infeasible", "bad_norm_dual", "bad_point", "missing_split"),
)
def test_numerical_failures_do_not_supply_false_minimum_or_unreachability(
    cvxpy, monkeypatch, fault
):
    original = backend.PreparedDistance.solve

    def broken(self, rhs):
        point, dual, report = original(self, rhs)
        if fault in ("solver_error", "infeasible"):
            return None, None, {**report, "status": fault}
        if fault == "bad_point":
            return [100] * len(point), None, report
        if fault == "missing_split":
            return point, (dual[0], dual[1], []), report
        return point, (0, [1e200] * len(point), [[0] * len(a) for a in dual[2]]), report

    monkeypatch.setattr(backend.PreparedDistance, "solve", broken)
    result = claim_for(requirements=("steam_header",)).boundary_result(backend="cvxpy")
    assert result.payload.search.resolution == "bounded"  # nosec B101
    assert (
        result.payload.search.lower.value
        <= 0.75
        <= result.payload.search.upper.value + 1e-14
    )  # nosec B101
    portable(result)


def test_unresolved_requirement_branch_retains_zero_global_lower_bound(
    cvxpy, monkeypatch
):
    original = backend.ConvexThresholdProblem.__init__

    def broken(self, claim, requirement, threshold, **kwargs):
        if requirement.name == "boiler_fuel":
            raise ValueError("Injected failed branch")
        original(self, claim, requirement, threshold, **kwargs)

    monkeypatch.setattr(backend.ConvexThresholdProblem, "__init__", broken)
    result = claim_for().boundary_result(backend="cvxpy")
    assert (
        result.payload.search.resolution == "bounded"
        and result.payload.search.lower.value == 0
    )  # nosec B101


def test_backend_requirements_and_nonconvex_domains_remain_explicit(monkeypatch):
    claim = claim_for()
    euclidean = replace(claim, distance=NormalizedL2(claim.domain.space))
    assert euclidean.boundary_result().execution == "unsupported"  # nosec B101
    assert claim.boundary_result(backend="unknown").execution == "invalid"  # nosec B101
    union = replace(claim, domain=Union((claim.domain, claim.domain)))
    assert union.boundary_result(backend="cvxpy").execution == "unsupported"  # nosec B101

    def unavailable(*a):
        raise ImportError("Injected missing optional dependency")

    monkeypatch.setattr(cvxpy_backend, "import_module", unavailable)
    assert euclidean.boundary_result(backend="cvxpy").execution == "unsupported"  # nosec B101


@pytest.mark.parametrize(
    "limit, constant, expected", ((0, None, "minimum_verified"), (1, 0, "unreachable"))
)
def test_zero_distance_and_constant_unreachable_targets(
    cvxpy, limit, constant, expected
):
    claim = scalar_claim(limit, constant=constant)
    result = replace(claim, distance=NormalizedL2(claim.domain.space)).boundary_result(
        backend="cvxpy"
    )
    assert result.payload.search.resolution == expected  # nosec B101
    portable(result)
