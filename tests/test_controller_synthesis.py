"""Controller synthesis checks permissions, complete coverage and actual commands."""

from dataclasses import replace
from fractions import Fraction as F
import json

import pytest

from operange import (
    AffineControlRule,
    AffineController,
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    ControlTarget,
    ControlTrackingObjective,
    ConvexHullSet,
    DecisionRule,
    FiniteSet,
    FrozenController,
    LinearObjective,
    NormalizedLInf,
    Parameter,
    RecoursePolicy,
    Scenario,
    result_from_json,
    synthesize_controller,
)
from operange import _controller_synthesis as synthesis
from examples.frozen_controllers import example
from tests.test_linear_process import small_model


@pytest.fixture
def cp():
    return pytest.importorskip("cvxpy")


def boiler(kind="box"):
    model, _ = example()
    box = BoxSet(
        (
            Parameter("dryer", "MW", 10, 8, 12, 2, "Test"),
            Parameter("evaporator", "MW", 6, 4, 8, 2, "Test"),
        )
    )
    if kind == "box":
        domain = box
    else:
        scenarios = tuple(
            Scenario(str(i), {"dryer": x, "evaporator": y}, "Test")
            for i, (x, y) in enumerate(((8, 4), (8, 8), (12, 4), (12, 8)))
        )
        domain = (
            FiniteSet(box.space, scenarios)
            if kind == "finite"
            else ConvexHullSet(box.space, scenarios)
        )
    return replace(model, objective=LinearObjective("fuel")).as_claim(
        domain, distance=NormalizedLInf(domain.space)
    )


def objective(result):
    return next(e.details for e in result.evidence if e.evidence_id == "objective")


@pytest.mark.parametrize("kind", ("box", "finite", "hull"))
def test_minimax_fuel_has_checked_performance_and_portable_replay(cp, kind):
    claim = boiler(kind)
    original = claim.contract
    result = claim.synthesize_controller(name="Minimum worst-case fuel")
    assert result.verified and result.execution == "completed"  # nosec B101
    assert claim.contract == original  # nosec B101
    assert result.claim.recourse == claim.recourse  # nosec B101
    assert result.claim.distance == claim.distance  # nosec B101
    assert result.audit.payload.coverage.method == (
        "complete_finite" if kind == "finite" else "analytical_domain"
    )  # nosec B101
    bound = objective(result)
    assert bound["lower"] <= 31 <= bound["upper"]  # nosec B101
    assert bound["upper"] - bound["lower"] < 1e-7  # nosec B101
    assert bound["optimality"] == "not_certified"  # nosec B101
    frozen = result.freeze()
    loaded = FrozenController.from_json(frozen.to_json())
    assert loaded.audit_result() == result.audit  # nosec B101
    assert loaded.contract == result.claim.contract  # nosec B101
    for compact in (False, True):
        exported = json.loads(result.to_json(compact=compact))
        assert result_from_json(json.dumps(exported["audit"])) == result.audit  # nosec B101
        assert (
            AffineController.from_json(json.dumps(exported["controller"]))
            == result.controller
        )  # nosec B101
        assert exported["source_contract"] == original.to_dict()  # nosec B101
        assert exported["synthesis_id"] == result.ref.artifact_id  # nosec B101


def test_partial_observations_never_become_full_observations(cp):
    claim = boiler()
    permissions = RecoursePolicy(
        "static",
        (
            DecisionRule("boiler_a", "MW", "operation", ("dryer",)),
            DecisionRule("boiler_b", "MW", "operation", ("evaporator",)),
        ),
    )
    claim = replace(claim, recourse=permissions)
    assert not claim.capabilities.audit.supported  # nosec B101
    result = claim.synthesize_controller()
    assert result.verified  # nosec B101
    for rule in result.controller.rules:
        assert set(rule.observes) <= set(permissions.rule(rule.control).observes)  # nosec B101
        permitted = {n: 10 if n == "dryer" else 6 for n in rule.observes}
        assert rule.command(permitted) == rule.command(
            {**permitted, "irrelevant": 1000}
        )  # nosec B101
    assert result.freeze().claim.recourse == permissions  # nosec B101


def test_no_observations_cannot_meet_variable_exact_load_even_when_adjustable_passes(
    cp,
):
    claim = boiler()
    assert claim.audit_result().payload.verdict == "pass"  # nosec B101
    claim = replace(
        claim,
        recourse=RecoursePolicy(
            "static",
            tuple(
                DecisionRule(c.name, c.unit, "operation")
                for c in claim.adapter.controls
            ),
        ),
    )
    result = claim.synthesize_controller()
    assert result.execution == "unresolved" and not result.verified  # nosec B101
    assert "does not prove" in result.diagnostics[0].message  # nosec B101
    with pytest.raises(ValueError, match="verified"):
        result.freeze()
    with pytest.raises(ValueError, match="candidate"):
        result.claim


@pytest.mark.parametrize("fixed", (True, False))
def test_fixed_and_zero_span_controls_remain_constants(cp, fixed):
    claim = boiler()
    if fixed:
        claim = replace(
            claim,
            recourse=RecoursePolicy(
                "static",
                (
                    DecisionRule("boiler_a", "MW", "operation", fixed_value=12),
                    claim.recourse.rule("boiler_b"),
                ),
            ),
        )
    else:
        claim = replace(
            claim,
            adapter=replace(
                claim.adapter,
                controls=(
                    replace(claim.adapter.controls[0], lower=12),
                    claim.adapter.controls[1],
                ),
            ),
        )
    result = claim.synthesize_controller()
    assert result.verified  # nosec B101
    rule = next(r for r in result.controller.rules if r.control == "boiler_a")
    assert rule.offset == 12 and rule.terms == ()  # nosec B101


def test_all_fixed_and_constant_domain_need_no_adjustable_coefficients(cp):
    model = small_model()
    box = BoxSet((Parameter("load", "MW", 0.5, 0.5, 0.5, 1, "Test"),))
    policy = RecoursePolicy(
        "fixed", (DecisionRule("power", "MW", "operation", fixed_value=0.5),)
    )
    result = model.as_claim(box, recourse=policy).synthesize_controller(
        objective=LinearObjective("margin")
    )
    assert result.verified and objective(result)["guaranteed_value"] == 0  # nosec B101
    assert not result.controller.rules[0].observes  # nosec B101


def test_maximize_guaranteed_margin_and_query_objective_do_not_change_model(cp):
    model = small_model(bounds=(0, 2))
    box = BoxSet((Parameter("load", "MW", 0.5, 0, 1, 0.5, "Test"),))
    claim = model.as_claim(box)
    result = synthesize_controller(
        claim, objective=LinearObjective("margin", "maximize")
    )
    assert result.verified and model.objective is None  # nosec B101
    assert objective(result)["aggregation"] == "minimum"  # nosec B101
    assert objective(result)["guaranteed_value"] == 1  # nosec B101
    assert result.controller.commands({"load": 1}) == {"power": 2}  # nosec B101
    # Choosing coefficients on an already bound claim is explicit retuning.
    revised = result.claim.synthesize_controller(objective=LinearObjective("margin"))
    assert revised.verified and objective(revised)["guaranteed_value"] == 0  # nosec B101
    assert revised.source.contract == result.claim.contract  # nosec B101


@pytest.mark.parametrize("kind", ("finite", "box", "hull"))
def test_tracking_is_worst_case_and_encloses_actual_interior_commands(cp, kind):
    claim = boiler(kind)
    target = ControlTrackingObjective(
        (ControlTarget("boiler_a", 10, 1, "MW"), ControlTarget("boiler_b", 6, 1, "MW"))
    )
    result = claim.synthesize_controller(objective=target)
    assert result.verified  # nosec B101
    bounds = objective(result)
    assert bounds["upper"] == pytest.approx(8, abs=1e-5)  # nosec B101
    scenarios = (
        [s.values for s in claim.domain.scenarios]
        if kind == "finite"
        else [
            {"dryer": 8 + i / 5, "evaporator": 4 + j / 5}
            for i in range(21)
            for j in range(21)
        ]
    )
    for point in scenarios:
        commands = result.controller.commands(point)
        value = sum(
            F(t.weight) * ((F(commands[t.control]) - F(t.target)) / F(t.scale)) ** 2
            for t in target.targets
        )
        assert value <= F(bounds["upper_exact"])  # nosec B101
    assert bounds["optimality"] == "not_certified"  # nosec B101


def test_zero_tolerance_identity_is_executable_on_continuous_domain(cp):
    model = small_model(
        limits=(AffineRequirement("no_excess", "margin", 0, tolerance=0),)
    )
    domain = BoxSet((Parameter("load", "MW", 0.5, 0, 1, 0.5, "Test"),))
    result = model.as_claim(domain).synthesize_controller(
        objective=LinearObjective("margin")
    )
    assert result.verified  # nosec B101
    assert result.controller.rules[0] == AffineControlRule(
        "power", "MW", 0, (AffineTerm("load", 1, "MW/MW"),)
    )  # nosec B101


def test_rounding_failure_inside_box_cannot_be_exported_as_verified(cp):
    model = small_model(
        bounds=(0, 2**54),
        limits=(AffineRequirement("no_excess", "margin", 0, tolerance=0),),
    )
    model = replace(model, outputs=(replace(model.outputs[0], offset=-(2**53)),))
    box = BoxSet((Parameter("load", "MW", 0, 0, 2, 1, "Rounding"),))
    result = model.as_claim(box).synthesize_controller(
        objective=LinearObjective("margin")
    )
    assert not result.verified  # nosec B101
    assert result.execution == "unresolved"  # nosec B101
    with pytest.raises(ValueError, match="verified"):
        result.freeze()


def test_requirement_selection_preserves_every_operating_and_control_limit(cp):
    claim = boiler()
    model = replace(
        claim.adapter,
        requirements=claim.adapter.requirements
        + (AffineRequirement("impossible_service", "fuel", -1),),
    )
    result = model.as_claim(
        claim.domain, requirements=("meet_load",)
    ).synthesize_controller()
    assert result.verified  # nosec B101
    assert result.claim.requirements == ("meet_load",)  # nosec B101
    too_small = replace(
        model,
        operating_limits=tuple(
            replace(r, limit=30) if r.name == "shared_fuel" else r
            for r in model.operating_limits
        ),
    )
    failed = too_small.as_claim(
        claim.domain, requirements=("meet_load",)
    ).synthesize_controller()
    assert failed.execution == "unresolved"  # nosec B101


@pytest.mark.parametrize(
    "options",
    (
        {"max_vertices": 0},
        {"max_vertices": True},
        {"name": ""},
        {"objective": {}},
        {"objective": LinearObjective("unknown")},
        {
            "objective": ControlTrackingObjective(
                (ControlTarget("boiler_a", 1, 1, "kg"),)
            )
        },
    ),
)
def test_invalid_requests_do_not_run_solver(monkeypatch, options):
    def no_backend():
        raise AssertionError("backend should not be loaded")

    monkeypatch.setattr(synthesis, "load_cvxpy", no_backend)
    result = boiler().synthesize_controller(**options)
    assert result.execution == "invalid" and not result.verified  # nosec B101
    json.loads(result.to_json())


@pytest.mark.parametrize("kind", ("box", "finite", "hull"))
def test_generator_budget_checked_before_solving(monkeypatch, kind):
    def no_backend():
        raise AssertionError("backend should not be loaded")

    monkeypatch.setattr(synthesis, "load_cvxpy", no_backend)
    result = boiler(kind).synthesize_controller(max_vertices=3)
    assert result.execution == "unsupported"  # nosec B101
    assert "4 generators" in result.diagnostics[0].message  # nosec B101
    assert result.controller is None and result.audit is None  # nosec B101


def test_duplicate_scenario_labels_count_toward_limit(cp):
    claim = boiler("finite")
    domain = replace(
        claim.domain,
        scenarios=claim.domain.scenarios
        + (replace(claim.domain.scenarios[0], name="duplicate point"),),
    )
    result = replace(claim, domain=domain).synthesize_controller(max_vertices=5)
    assert result.verified  # nosec B101
    assert result.evidence[0].details["generator_count"] == 5  # nosec B101
    assert (
        replace(claim, domain=domain).synthesize_controller(max_vertices=4).execution
        == "unsupported"
    )  # nosec B101


def test_hull_without_normalization_metadata_uses_exact_generator_extrema(cp):
    model, domain = example()
    hull = ConvexHullSet(domain.space, domain.scenarios)
    result = model.as_claim(hull).synthesize_controller(
        objective=LinearObjective("fuel")
    )
    assert result.verified  # nosec B101
    assert objective(result)["guaranteed_value"] == pytest.approx(31)  # nosec B101
    enclosure = next(
        e.details for e in result.audit.evidence if e.evidence_id == "enclosure"
    )
    assert all(
        p["method"] == "exact_hull_affine_extremum" for p in enclosure["support_bounds"]
    )  # nosec B101
    assert (
        FrozenController.from_json(result.freeze().to_json()).audit_result()
        == result.audit
    )  # nosec B101


def test_missing_objective_and_invalid_permissions_are_rejected():
    claim = boiler()
    assert (
        replace(claim, adapter=replace(claim.adapter, objective=None))
        .synthesize_controller()
        .execution
        == "invalid"
    )  # nosec B101
    for rule in (
        replace(claim.recourse.rules[0], observes=("unknown",)),
        replace(claim.recourse.rules[0], stage="future"),
    ):
        invalid = replace(
            claim,
            recourse=replace(claim.recourse, rules=(rule, claim.recourse.rules[1])),
        )
        assert invalid.synthesize_controller().execution == "invalid"  # nosec B101
    incomplete = replace(
        claim, recourse=replace(claim.recourse, rules=(claim.recourse.rules[0],))
    )
    assert incomplete.synthesize_controller().execution == "invalid"  # nosec B101


def test_unsupported_domains_are_never_sampled():
    from operange import Union, PolytopeSet, LinearConstraint

    claim = boiler()
    for domain in (
        Union((claim.domain, claim.domain)),
        PolytopeSet(
            claim.domain, (LinearConstraint("sum", {"dryer": 1, "evaporator": 1}, 1),)
        ),
    ):
        result = replace(claim, domain=domain).synthesize_controller()
        assert result.execution == "unsupported"  # nosec B101
        assert result.audit is None and result.controller is None  # nosec B101


def test_new_domain_requires_new_audit(cp):
    result = boiler().synthesize_controller()
    frozen = result.freeze()
    enlarged = BoxSet(
        (
            Parameter("dryer", "MW", 10, 8, 14, 2, "Enlarged"),
            Parameter("evaporator", "MW", 6, 4, 8, 2, "Enlarged"),
        )
    )
    audit = frozen.as_claim(enlarged).audit_result()
    assert audit.payload.verdict == "fail"  # nosec B101
    assert frozen.audit_result().payload.verdict == "pass"  # nosec B101


def test_missing_dependency_is_explicit_and_freezing_does_not_load_it(cp, monkeypatch):
    from operange import _cvxpy_backend

    result = boiler().synthesize_controller()

    def unavailable(*args):
        raise ImportError("Injected missing dependency")

    monkeypatch.setattr(_cvxpy_backend, "import_module", unavailable)
    failed = boiler().synthesize_controller()
    assert failed.execution == "unsupported"  # nosec B101
    assert "operange[cvxpy]" in failed.diagnostics[0].message  # nosec B101
    loaded = FrozenController.from_json(result.freeze().to_json())
    assert loaded.audit_result().payload.verdict == "pass"  # nosec B101


@pytest.mark.parametrize("status", ("infeasible", "unbounded", "unresolved"))
def test_solver_status_alone_has_no_engineering_verdict(cp, monkeypatch, status):
    monkeypatch.setattr(
        synthesis.PolicyProgram, "solve", lambda *args: (None, {"status": status})
    )
    result = boiler().synthesize_controller()
    assert result.execution == "unresolved" and result.audit is None  # nosec B101
    assert result.controller is None and not result.verified  # nosec B101


def test_bad_solver_candidate_is_reaudited_without_clipping_or_redispatch(
    cp, monkeypatch
):
    bad = AffineController(
        "Bad proposal",
        (
            AffineControlRule("boiler_a", "MW", 99),
            AffineControlRule("boiler_b", "MW", 99),
        ),
        "Injected",
    )
    monkeypatch.setattr(synthesis.PolicyProgram, "controller", lambda *args: bad)
    result = boiler().synthesize_controller()
    assert result.execution == "unresolved"  # nosec B101
    assert result.audit.payload.verdict == "fail"  # nosec B101
    assert result.audit.payload.witness.kind == "fixed_policy_failure"  # nosec B101
    assert result.controller.commands({}) == {"boiler_a": 99, "boiler_b": 99}  # nosec B101
    with pytest.raises(ValueError, match="verified"):
        result.freeze()


def test_solver_exception_and_nonfinite_candidate_remain_unresolved(cp, monkeypatch):
    monkeypatch.setattr(
        synthesis.PolicyProgram,
        "solve",
        lambda *args: (
            {"boiler_a": [float("nan"), 0, 0], "boiler_b": [0, 0, 0]},
            {"status": "unresolved"},
        ),
    )
    result = boiler().synthesize_controller()
    assert result.execution == "unresolved"  # nosec B101
    json.loads(result.to_json())

    def fail(*args):
        raise RuntimeError("Injected backend error")

    monkeypatch.setattr(synthesis.PolicyProgram, "solve", fail)
    assert boiler().synthesize_controller().execution == "unresolved"  # nosec B101


def test_unexportable_output_prevents_verified_candidate(cp):
    claim = boiler()
    extra = AffineOutput(
        "overflow",
        "MW",
        "thermal_power",
        0,
        (AffineTerm("dryer", 1e308, "MW/MW"),),
        "Test",
    )
    claim = replace(
        claim, adapter=replace(claim.adapter, outputs=claim.adapter.outputs + (extra,))
    )
    result = claim.synthesize_controller()
    assert result.execution == "unresolved"  # nosec B101
    assert result.audit.payload.verdict == "inconclusive"  # nosec B101


def test_physical_coefficients_keep_units_negative_slopes_and_offsets(cp):
    from operange import LinearControl, LinearProcessAdapter

    domain = BoxSet((Parameter("load", "kW", 1500, 1000, 2000, 500, "Test"),))
    model = LinearProcessAdapter(
        "Mixed-unit complementary demand",
        domain.space,
        (
            AffineOutput(
                "balance",
                "MW",
                "thermal_power",
                -3,
                (AffineTerm("power", 1, "MW/MW"), AffineTerm("load", 0.001, "MW/kW")),
                "Complementary supply",
            ),
        ),
        (AffineRequirement("service", "balance", 0, "ge"),),
        (LinearControl("power", "MW", -3, 5),),
        (AffineRequirement("no_excess", "balance", 0),),
        objective=LinearObjective("balance"),
    )
    result = model.as_claim(domain).synthesize_controller()
    assert result.verified  # nosec B101
    rule = result.controller.rules[0]
    assert rule.offset == pytest.approx(3)  # nosec B101
    assert rule.terms[0].unit == "MW/kW"  # nosec B101
    assert rule.terms[0].coefficient == pytest.approx(-0.001)  # nosec B101
    assert (
        result.claim.evaluate_result({"load": 1432.125}).payload.feasibility
        == "feasible"
    )  # nosec B101


def test_independent_objective_check_failure_keeps_passing_audit_but_no_export(
    cp, monkeypatch
):
    def overflow(*args):
        raise OverflowError("Injected objective export failure")

    monkeypatch.setattr(synthesis, "_objective_bound", overflow)
    result = boiler().synthesize_controller()
    assert result.audit.payload.verdict == "pass"  # nosec B101
    assert result.execution == "unresolved" and not result.verified  # nosec B101
    with pytest.raises(ValueError, match="verified"):
        result.freeze()


def test_consumer_example_includes_verified_replay_and_unsuccessful_policy_search(cp):
    from examples.controller_synthesis import run_example

    exported = run_example()
    assert exported["fuel"]["verified"] and exported["tracking"]["verified"]  # nosec B101
    assert (
        result_from_json(json.dumps(exported["interior"])).payload.feasibility
        == "feasible"
    )  # nosec B101
    assert exported["constant_policy_search"]["execution"] == "unresolved"  # nosec B101
