"""Physical actuators have policies; algebraic states have equations and bounds."""

from dataclasses import replace
from fractions import Fraction as F

import pytest

from operange import (
    AffineControlRule,
    AffineController,
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    DecisionRule,
    FiniteSet,
    FrozenController,
    LinearControl,
    LinearState,
    LinearObjective,
    LinearProcessAdapter,
    OperatingSpec,
    Parameter,
    RecoursePolicy,
    Scenario,
    verify_result,
)
from operange.dispatch_sensitivity import DispatchChange


def boiler():
    domain = BoxSet((Parameter("load", "MW", 3, 2, 4, 1, "Demand"),))

    def output(name, **terms):
        return AffineOutput(
            name,
            "MW",
            "thermal_power",
            0,
            tuple(AffineTerm(n, a, "MW/MW") for n, a in terms.items()),
            "Test boiler",
        )

    model = LinearProcessAdapter(
        "Boiler with separate actuator and steam state",
        domain.space,
        (
            output("balance", steam=1, fuel=-0.5),
            output("delivery", steam=1, load=-1),
            output("fuel_use", fuel=1),
        ),
        (AffineRequirement("service", "delivery", 0, "ge"),),
        (LinearControl("fuel", "MW", 0, 10),),
        (
            AffineRequirement("balance:le", "balance", 0),
            AffineRequirement("balance:ge", "balance", 0, "ge"),
        ),
        objective=LinearObjective("fuel_use", tolerance=1e-7),
        states=(LinearState("steam", "MW", 0, 6),),
    )
    cases = FiniteSet(
        domain.space,
        tuple(Scenario(str(n), {"load": n}, "Test demand") for n in (2, 3, 4)),
    )
    return model, domain, cases


def values(result):
    return {v.quantity_ref: v.value for v in result.payload.values}


@pytest.mark.parametrize("backend", ("scipy", "cvxpy"))
def test_dispatch_and_fixed_actuators_solve_states_and_verify(backend):
    if backend == "cvxpy":
        pytest.importorskip("cvxpy")
    model, domain, _ = boiler()
    spec = OperatingSpec.fixed(model.controls, {"fuel": 8})
    claim = model.as_claim(domain, operating=spec)
    assert tuple(r.control for r in claim.recourse.rules) == ("fuel",)  # nosec B101
    assert claim.contract.quantity("steam").role == "state"  # nosec B101
    assert "state_upper:steam" in {c.constraint_id for c in claim.contract.constraints}  # nosec B101
    result = claim.evaluate_result(domain.nominal, backend=backend)
    assert result.payload.feasibility == "feasible", result.to_json()  # nosec B101
    assert values(result)["fuel"] == 8 and values(result)["steam"] == pytest.approx(4)  # nosec B101
    assert verify_result(result).verified  # nosec B101
    audit = claim.audit_result(backend=backend)
    assert audit.payload.verdict == "pass" and verify_result(audit).verified  # nosec B101
    optimized = model.as_claim(domain).evaluate_result(domain.nominal, backend=backend)
    assert optimized.payload.objective.optimality == "verified"  # nosec B101
    assert values(optimized)["fuel"] == pytest.approx(6, abs=1e-6)  # nosec B101
    assert verify_result(optimized).verified  # nosec B101


def test_roles_cannot_be_overridden_by_permissions_objectives_or_controller_rules():
    model, _, cases = boiler()
    with pytest.raises(ValueError, match="role"):
        model.as_claim(
            cases,
            recourse=RecoursePolicy(
                "fixed",
                (
                    DecisionRule("fuel", "MW", "operation", fixed_value=8),
                    DecisionRule("steam", "MW", "operation", fixed_value=4),
                ),
            ),
        )
    with pytest.raises(ValueError, match="control"):
        model.as_claim(
            cases,
            controller=AffineController(
                "Wrong",
                (
                    AffineControlRule("fuel", "MW", 8),
                    AffineControlRule("steam", "MW", 4),
                ),
                "A state is not an actuator",
            ),
        )
    with pytest.raises(ValueError, match="distinct"):
        replace(model, states=(LinearState("fuel", "MW", 0, 10),))
    with pytest.raises(ValueError, match="operating"):
        model.as_claim(
            cases,
            operating=OperatingSpec.fixed(model.controls, {"fuel": 8}),
            recourse=model.as_claim(cases).recourse,
        )
    shared = model.as_claim(cases, operating=OperatingSpec.shared(model.controls))
    assert shared.evaluate_result({"load": 3}).execution == "unsupported"  # nosec B101


@pytest.mark.parametrize("command, verdict", ((8, "pass"), (4, "fail"), (12, "fail")))
def test_controller_commands_only_actuators_and_checks_joint_state_feasibility(
    command, verdict
):
    model, domain, cases = boiler()
    controller = AffineController(
        "Saved fuel command",
        (AffineControlRule("fuel", "MW", command),),
        "Test command",
    )
    claim = model.as_claim(cases, operating=OperatingSpec.controlled(controller))
    audit = claim.audit_result()
    assert audit.payload.verdict == verdict, audit.to_json()  # nosec B101
    assert verify_result(audit).verified, verify_result(audit).to_dict()  # nosec B101
    for result in audit.supporting_evaluations:
        assert values(result)["fuel"] == command  # nosec B101
        assert verify_result(result).verified  # nosec B101
        if result.payload.feasibility == "feasible":
            assert values(result)["steam"] == pytest.approx(command / 2)  # nosec B101
        else:
            assert "steam" not in values(result)  # nosec B101
            assert all(
                c.assessment == "unknown" for c in result.payload.constraint_checks
            )  # nosec B101
    frozen = FrozenController.from_json(claim.freeze().to_json())
    assert frozen.contract.ref == claim.contract.ref  # nosec B101
    assert frozen.audit_result().payload.verdict == verdict  # nosec B101
    assert frozen.as_claim(domain).audit_result().execution == "unsupported"  # nosec B101


def test_synthesis_has_actuator_rules_and_scenario_states():
    pytest.importorskip("cvxpy")
    model, domain, cases = boiler()
    synthesized = model.as_claim(cases).synthesize_controller()
    assert synthesized.verified, synthesized.to_json()  # nosec B101
    assert tuple(r.control for r in synthesized.controller.rules) == ("fuel",)  # nosec B101
    assert verify_result(synthesized.audit).verified  # nosec B101
    shared = model.as_claim(
        cases, operating=OperatingSpec.shared(model.controls)
    ).synthesize_controller()
    assert shared.verified and not shared.controller.rules[0].terms  # nosec B101
    assert shared.controller.rules[0].offset == pytest.approx(8, abs=1e-6)  # nosec B101
    assert verify_result(shared.audit).verified  # nosec B101
    replay = synthesized.freeze().evaluate_result({"load": 3})
    assert replay.payload.feasibility == "feasible" and verify_result(replay).verified  # nosec B101
    assert model.as_claim(domain).synthesize_controller().execution == "unsupported"  # nosec B101
    dependent_cost = replace(model, objective=LinearObjective("delivery"))
    assert (
        dependent_cost.as_claim(cases).synthesize_controller().execution
        == "unsupported"
    )  # nosec B101


def test_state_bounds_and_equations_remain_in_relief_and_local_objective_proofs():
    model, domain, _ = boiler()
    limited = replace(model, states=(replace(model.states[0], upper=2.5),))
    result = limited.as_claim(domain).evaluate_result({"load": 3}, diagnose=True)
    assert result.payload.feasibility == "infeasible" and verify_result(result).verified  # nosec B101
    conflict = next(e for e in result.evidence if e.evidence_id == "conflict")
    assert "state_upper:steam" in conflict.details["certificate_bound_refs"]  # nosec B101
    sensitivity = model.as_claim(domain).dispatch_sensitivity(
        {"load": 3}, changes=(DispatchChange("input", "load", "MW", 0.1),)
    )
    assert verify_result(sensitivity).verified, sensitivity.to_json()  # nosec B101
    slope = sensitivity.sensitivities[0]
    assert F(slope.backward.slope_exact) == F(slope.forward.slope_exact) == 2  # nosec B101
    capacity = model.as_claim(domain).dispatch_sensitivity(
        {"load": 3}, changes=(DispatchChange("state_upper", "steam", "MW", 0.1),)
    )
    assert verify_result(capacity).verified  # nosec B101
    wrong_role = model.as_claim(domain).dispatch_sensitivity(
        {"load": 3}, changes=(DispatchChange("control_upper", "steam", "MW", 0.1),)
    )
    assert wrong_role.execution == "invalid"  # nosec B101
    capped = replace(
        model,
        operating_limits=model.operating_limits
        + (AffineRequirement("fuel_cap", "fuel_use", 5),),
    )
    claim = capped.as_claim(domain)
    point = claim.evaluate_result(
        {"load": 3}, relief={"constraint": "fuel_cap", "maximum": 5}
    )
    assert verify_result(point).verified, verify_result(point).to_dict()  # nosec B101
    audit = claim.audit_result(
        relief={
            "objective": "linear",
            "changes": (
                {"constraint": "fuel_cap", "maximum": 5, "scale": 1, "unit": "MW"},
            ),
        }
    )
    assert verify_result(audit).verified, verify_result(audit).to_dict()  # nosec B101


def test_independent_checker_rejects_forged_state_values(monkeypatch):
    from operange import linear

    model, _, cases = boiler()
    claim = model.as_claim(
        cases,
        controller=AffineController(
            "Fuel", (AffineControlRule("fuel", "MW", 8),), "Test"
        ),
    )
    result = claim.evaluate_result({"load": 3})

    def no_solver(*args, **kwargs):
        raise AssertionError("independent verification must not solve a model")

    monkeypatch.setattr(linear, "solve_lp", no_solver)
    assert verify_result(result).verified  # nosec B101
    forged = replace(
        result,
        payload=replace(
            result.payload,
            values=tuple(
                replace(v, value=5) if v.quantity_ref == "steam" else v
                for v in result.payload.values
            ),
        ),
    )
    assert not verify_result(forged).verified  # nosec B101


def test_controller_state_solver_failure_stays_unresolved(monkeypatch):
    from operange import linear
    from operange.results import SolveEvidence

    model, _, cases = boiler()
    claim = model.as_claim(
        cases,
        controller=AffineController(
            "Fuel", (AffineControlRule("fuel", "MW", 4),), "Test"
        ),
    )
    monkeypatch.setattr(
        linear,
        "solve_lp",
        lambda *a, **k: (
            None,
            SolveEvidence("unresolved", "No candidate", "test", 1e-9),
        ),
    )
    result = claim.evaluate_result({"load": 3})
    assert result.execution == "unresolved" and result.payload.feasibility == "unknown"  # nosec B101
    audit = claim.audit_result()
    assert audit.payload.verdict == "inconclusive" and audit.payload.witness is None  # nosec B101


def test_explicit_network_actuators_and_passive_network_keep_portable_roles():
    from examples.equipment_network import example
    from tests.test_network import mixing
    from operange import ProcessNetwork

    network, domain = example()
    declared = replace(network, actuators=("fuel_a", "fuel_b"))
    model = ProcessNetwork.from_json(declared.to_json()).compile()
    assert {c.name for c in model.controls} == {"fuel_a", "fuel_b"}  # nosec B101
    assert {s.name for s in model.states} == {s.name for s in network.streams} - {
        "fuel_a",
        "fuel_b",
    }  # nosec B101
    audit = model.as_claim(domain).audit_result()
    assert audit.payload.verdict == "pass" and verify_result(audit).verified  # nosec B101
    legacy = ProcessNetwork.from_json(network.to_json())
    assert legacy.ref == network.ref and not legacy.compile().states  # nosec B101
    passive, inputs = mixing()
    passive_model = replace(passive, actuators=()).compile()
    assert not passive_model.controls and passive_model.states  # nosec B101
    audit = passive_model.as_claim(inputs).audit_result()
    assert audit.payload.verdict == "pass" and verify_result(audit).verified  # nosec B101
