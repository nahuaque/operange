"""Request-specific support is discoverable without generating solver candidates."""

from dataclasses import FrozenInstanceError, replace

import pytest

from operange import (
    AuditRequest,
    BoundaryRequest,
    BreakingRequest,
    ControllerSynthesisRequest,
    DispatchChange,
    DispatchSensitivityRequest,
    EvaluationRequest,
    OperatingLimitRequest,
    OperatingSpec,
    SensitivityRequest,
    StudyRequestError,
    ControlTarget,
    ControlTrackingObjective,
    NormalizedL2,
    NormalizedLInf,
    check_study,
    run_study,
)
from examples.controller_synthesis import example as boiler_claim
from examples.steam_header import example as header
from tests.test_variable_roles import boiler as state_boiler


def boiler():
    claim = boiler_claim()
    return claim.adapter, claim.domain


@pytest.mark.parametrize(
    "study",
    (
        EvaluationRequest({"dryer": 10, "evaporator": 6}),
        AuditRequest(),
        OperatingLimitRequest("dryer", max_iterations=0),
        DispatchSensitivityRequest(
            {"dryer": 10, "evaporator": 6},
            (DispatchChange("input", "dryer", "MW", 0.1),),
        ),
    ),
)
def test_explicit_requests_preserve_existing_result_families_and_artifacts(study):
    model, domain = boiler()
    claim = model.as_claim(domain)
    cap = claim.check_study(study)
    assert cap.supported and "independent_certificate" in cap.evidence  # nosec B101
    if isinstance(study, EvaluationRequest):
        before = claim.evaluate_result(study.realization)
    elif isinstance(study, AuditRequest):
        before = claim.audit_result()
    elif isinstance(study, OperatingLimitRequest):
        before = claim.operating_limit(study.parameter, max_iterations=0)
    else:
        before = claim.dispatch_sensitivity(study.realization, changes=study.changes)
    after = claim.run_study(study)
    assert type(after) is type(before) and after.to_json() == before.to_json()  # nosec B101


def test_fixed_response_and_distance_requests_keep_their_existing_meaning():
    model, domain = header()
    claim = model.as_claim(domain, distance=NormalizedLInf(domain.space))
    for study, result in (
        (
            SensitivityRequest(domain.nominal, outputs=("steam",), inputs=("dryer",)),
            claim.sensitivity_result(
                domain.nominal, outputs=("steam",), inputs=("dryer",)
            ),
        ),
        (BoundaryRequest(), claim.boundary_result()),
        (
            BreakingRequest(
                violation_margins={r.name: 0.01 for r in model.requirements}
            ),
            claim.breaking_result(
                violation_margins={r.name: 0.01 for r in model.requirements}
            ),
        ),
    ):
        assert check_study(claim, study).supported  # nosec B101
        assert run_study(claim, study).to_json() == result.to_json()  # nosec B101
    euclidean = replace(claim, distance=NormalizedL2(domain.space))
    report = euclidean.check_study(BoundaryRequest())
    assert report.status == "unsupported" and "cvxpy" in report.scope  # nosec B101
    assert (
        claim.check_study(
            SensitivityRequest(domain.nominal, response="optimized")
        ).status
        == "unsupported"
    )  # nosec B101


def test_preflight_does_not_solve_replay_or_enumerate(monkeypatch):
    from operange import (
        Claim,
        linear,
        _controller_synthesis,
        _cvxpy_backend,
        _domain_generators,
    )
    from operange import _finite_audit
    from operange import _study_dispatch

    model, domain = boiler()
    claim = model.as_claim(domain)
    shared = model.as_claim(domain, operating=OperatingSpec.shared(model.controls))

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight executed a study")

    monkeypatch.setattr(linear, "solve_lp", forbidden)
    monkeypatch.setattr(Claim, "_run", forbidden)
    monkeypatch.setattr(_cvxpy_backend, "load_cvxpy", forbidden)
    monkeypatch.setattr(_controller_synthesis, "_generators", forbidden)
    monkeypatch.setattr(_domain_generators, "generators", forbidden)
    monkeypatch.setattr(_finite_audit, "audit_finite", forbidden)
    # Dependency discovery is separate from formulation eligibility.
    monkeypatch.setattr(_study_dispatch, "find_spec", lambda name: object())
    for study in (
        EvaluationRequest(domain.nominal),
        AuditRequest(),
        DispatchSensitivityRequest(
            domain.nominal, (DispatchChange("input", "dryer", "MW", 0.1),)
        ),
        OperatingLimitRequest("dryer"),
        ControllerSynthesisRequest(max_vertices=1),
    ):
        assert claim.check_study(study).supported  # nosec B101
    assert not shared.capabilities.audit.supported  # nosec B101
    assert shared.check_study(ControllerSynthesisRequest()).supported  # nosec B101
    assert shared.check_study(AuditRequest()).status == "unsupported"  # nosec B101
    assert claim.check_study(AuditRequest(max_vertices=1)).status == "unsupported"  # nosec B101
    tracking = ControlTrackingObjective((ControlTarget("boiler_a", 10, 1, "MW"),))
    assert (
        claim.check_study(
            ControllerSynthesisRequest(objective=tracking, max_vertices=1)
        ).status
        == "unsupported"
    )  # nosec B101


def test_state_and_coverage_limits_are_visible_without_policy_search(monkeypatch):
    from operange import _study_dispatch

    monkeypatch.setattr(_study_dispatch, "find_spec", lambda name: object())
    model, box, cases = state_boiler()
    study = ControllerSynthesisRequest()
    finite = model.as_claim(cases).check_study(study)
    continuous = model.as_claim(box).check_study(study)
    assert finite.supported and finite.coverage == "finite_scenarios"  # nosec B101
    assert "continuous_coverage" not in finite.evidence  # nosec B101
    assert continuous.status == "unsupported" and "Algebraic-state" in continuous.scope  # nosec B101
    cap = model.as_claim(cases).check_study(
        AuditRequest(required_evidence=("continuous_coverage",))
    )
    assert not cap.supported and cap.coverage == "finite_scenarios"  # nosec B101


def test_dependency_discovery_and_evidence_requirements_prevent_execution(monkeypatch):
    from operange import _study_dispatch

    model, domain = boiler()
    claim = model.as_claim(domain)
    real_find = _study_dispatch.find_spec
    monkeypatch.setattr(
        _study_dispatch,
        "find_spec",
        lambda name: None if name == "cvxpy" else real_find(name),
    )
    report = claim.check_study(ControllerSynthesisRequest())
    assert not report.supported and report.dependencies == ("cvxpy",)  # nosec B101
    with pytest.raises(StudyRequestError) as error:
        claim.run_study(ControllerSynthesisRequest())
    assert error.value.capability == report  # nosec B101
    report = claim.check_study(
        ControllerSynthesisRequest(required_evidence=("global_optimality",))
    )
    assert not report.supported and "global_optimality" in report.scope  # nosec B101
    assert claim.check_study(AuditRequest()).supported  # nosec B101
    from operange import Intersection

    affine, loads = header()
    composed = affine.as_claim(Intersection((loads, loads), backend="cvxpy"))
    report = composed.check_study(AuditRequest())
    assert report.status == "unsupported" and report.dependencies == ("cvxpy",)  # nosec B101
    assert composed.check_study(EvaluationRequest(loads.nominal)).supported  # nosec B101


@pytest.mark.parametrize(
    "study",
    (
        EvaluationRequest({"dryer": 10}),
        EvaluationRequest({"dryer": 10, "evaporator": 6}, diagnose=1),
        AuditRequest(max_vertices=0),
        AuditRequest(max_vertices=True),
        AuditRequest(backend="typo"),
        AuditRequest(relief={"constraint": "shared_fuel", "maximum": 1}),
        DispatchSensitivityRequest({"dryer": 10, "evaporator": 6}, ()),
        OperatingLimitRequest("missing"),
        OperatingLimitRequest("dryer", max_iterations=-1),
        AuditRequest(required_evidence=("made_up",)),
        object(),
    ),
)
def test_malformed_requests_are_distinct_from_unsupported_methods(study):
    model, domain = boiler()
    claim = model.as_claim(domain)
    cap = claim.check_study(study)
    assert cap.status == "invalid", cap  # nosec B101
    with pytest.raises(StudyRequestError):
        claim.run_study(study)


def test_relief_capability_checks_physical_targets_objective_and_complete_budget():
    model, domain = boiler()
    claim = model.as_claim(domain)
    options = {
        "changes": (
            {"constraint": "shared_fuel", "maximum": 2, "scale": 1, "unit": "MW"},
        ),
        "objective": "linear",
    }
    study = AuditRequest(relief=options, required_evidence=("relief_bounds",))
    assert claim.check_study(study).supported  # nosec B101
    assert (
        claim.check_study(replace(study, relief={**options, "max_scenarios": 1})).status
        == "unsupported"
    )  # nosec B101
    assert (
        claim.check_study(
            replace(study, relief={**options, "objective": "quadratic"})
        ).status
        == "unsupported"
    )  # nosec B101
    options["changes"][0]["constraint"] = "meet_load"
    assert claim.check_study(AuditRequest(relief=options)).status == "invalid"  # nosec B101
    assert study.relief["changes"][0]["constraint"] == "shared_fuel"  # nosec B101
    with pytest.raises(TypeError):
        study.relief["objective"] = "quadratic"
    with pytest.raises(FrozenInstanceError):
        study.max_vertices = 1


def test_synthesis_request_executes_and_returns_its_own_result_type():
    pytest.importorskip("cvxpy")
    model, domain = boiler()
    claim = model.as_claim(domain, operating=OperatingSpec.shared(model.controls))
    result = claim.run_study(ControllerSynthesisRequest())
    # A permitted policy search can be unsuccessful; capability is not a verdict.
    assert result.execution in ("completed", "unresolved")  # nosec B101
    assert result.source is claim  # nosec B101
    assert claim.recourse == OperatingSpec.shared(model.controls).policy  # nosec B101
    dispatched = model.as_claim(domain).run_study(ControllerSynthesisRequest())
    assert dispatched.verified  # nosec B101


def test_consumer_owned_original_adapter_keeps_its_protocol():
    from tests.test_dsl import EvaluationOnlyAdapter
    from operange import BoxSet, Claim, DecisionRule, Parameter, RecoursePolicy

    domain = BoxSet((Parameter("feed", "kg/s", 1, 0, 2, 1, "Test"),))
    claim = Claim(
        EvaluationOnlyAdapter(),
        domain,
        RecoursePolicy(
            "static", (DecisionRule("outlet", "kg/s", "operation", ("feed",)),)
        ),
    )
    study = EvaluationRequest({"feed": 1})
    cap = claim.check_study(study)
    assert cap.supported and not cap.evidence and cap.coverage == "adapter_declared"  # nosec B101
    assert (
        claim.run_study(study).to_json() == claim.evaluate_result({"feed": 1}).to_json()
    )  # nosec B101
    assert not claim.check_study(
        replace(study, required_evidence=("independent_certificate",))
    ).supported  # nosec B101


def test_public_example_keeps_capability_and_outcome_separate():
    from examples.study_requests import run_example

    report = run_example()
    assert report["audit_capability"]["status"] == "supported"  # nosec B101
    assert report["audit"]["payload"]["verdict"] == "pass"  # nosec B101
    assert report["verification"]["status"] == "verified"  # nosec B101
    assert report["budget_capability"]["status"] == "unsupported"  # nosec B101
