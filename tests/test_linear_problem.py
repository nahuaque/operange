"""Production compilation agrees with native equations, not another compiler."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
from itertools import product

import pytest

from operange import (
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    ControlTarget,
    ControlTrackingObjective,
    LinearControl,
    LinearObjective,
    LinearProcessAdapter,
    LinearState,
    OperatingSpec,
    Parameter,
)
from operange._linear_compile import compile_model
from operange._linear_problem import (
    AffineExpression,
    LinearProblem,
    LinearSystem,
    PhysicalRow,
    Polynomial,
    RowSource,
    Variable,
    coordinates,
)
from operange._linear_transform import (
    ReliefColumn,
    augment_relief,
    relax_rows,
    stack_scenarios,
)
from operange._parametric_dispatch import change_rates, problem
from operange.dispatch_sensitivity import DispatchChange


def plant():
    domain = BoxSet((Parameter("load", "MW", 1, 0, 2, 1, "Test load"),))

    def output(name, offset, **terms):
        return AffineOutput(
            name,
            "MW",
            "thermal_power",
            offset,
            tuple(AffineTerm(n, a, "MW/MW") for n, a in terms.items()),
            "Test plant",
        )

    model = LinearProcessAdapter(
        "Native-unit compilation fixture",
        domain.space,
        (
            output("duty", 0.5, flow=1, feed=-0.5, trim=0.25, load=-0.5),
            output("delivery", -1, flow=1, load=-1, trim=0.5),
            output("value", 3, feed=2, trim=-0.5, flow=0.25, load=-1),
        ),
        (
            AffineRequirement(
                "service", "delivery", 0, "ge", tolerance=0.125, residual_scale=2
            ),
            AffineRequirement("excluded", "delivery", 100, "ge"),
        ),
        (LinearControl("feed", "MW", -2, 6), LinearControl("trim", "MW", 1, 1)),
        (
            AffineRequirement("hot", "duty", 2, tolerance=0.125, residual_scale=4),
            AffineRequirement(
                "cold", "duty", -2, "ge", tolerance=0.25, residual_scale=0.5
            ),
        ),
        objective=LinearObjective("value", "maximize"),
        states=(LinearState("flow", "MW", -3, 8),),
    )
    return model, domain


def output_value(model, name, values):
    declaration = model.output(name)
    return F(declaration.offset) + sum(
        (F(t.coefficient) * F(values[t.variable]) for t in declaration.terms), F(0)
    )


@pytest.mark.parametrize("fixed", ({}, {"feed": 2}, {"feed": 7}, {"trim": 2}))
def test_normalized_and_explicit_rows_match_native_feasibility(fixed):
    model, _ = plant()
    physical = compile_model(model, ("service",))
    limits = model.operating_limits + model.requirements[:1]
    names = tuple(v.name for v in model.variables)
    seen = set()
    for load in (0, 1.5):
        point = {"load": load}
        system = physical.normalized(point, fixed)
        explicit = [row.at(point, names) for row in physical.explicit_rows(fixed)]
        assert "excluded" not in system.row_refs  # nosec B101
        assert set(system.fixed) == {"trim"} | set(fixed)  # nosec B101
        for feed, flow in product((-3, -2, 0, 2, 6, 7), (-4, -3, 0, 3, 8, 9)):
            values = {"feed": feed, "trim": fixed.get("trim", 1), "flow": flow}
            expected = (
                all(v.lower <= values[v.name] <= v.upper for v in model.variables)
                and all(values[n] == value for n, value in fixed.items())
                and all(
                    r.sign
                    * (output_value(model, r.output, {**point, **values}) - F(r.limit))
                    <= F(r.tolerance)
                    for r in limits
                )
            )
            seen.add(expected)
            assert (system.checked_values(values) is not None) == expected  # nosec B101
            assert (
                all(
                    sum((a * F(values[n]) for n, a in zip(names, row)), F(0)) <= rhs
                    for row, rhs in explicit
                )
                == expected
            )  # nosec B101
    assert False in seen  # nosec B101
    assert (True in seen) == (not fixed or fixed == {"feed": 2})  # nosec B101


@pytest.mark.parametrize("tracking", (False, True))
@pytest.mark.parametrize("fixed", ({}, {"feed": 2}))
def test_normalized_objective_preserves_native_value_including_fixed_cost(
    tracking, fixed
):
    model, _ = plant()
    if tracking:
        model = replace(
            model,
            objective=ControlTrackingObjective(
                (
                    ControlTarget("feed", 1.5, 2, "MW", 0.5),
                    ControlTarget("trim", 0, 0.5, "MW", 2),
                )
            ),
        )
    physical = compile_model(model, ("service",))
    point = {"load": 1.5}
    system = physical.normalized(point, fixed)
    polynomial = physical.objective.normalized(system, point)
    for feed, flow in product((-2, 0, 2, 6), (-3, 0, 8)):
        values = {"feed": feed, "flow": flow, **system.fixed}
        expected = (
            sum(
                (
                    F(t.weight)
                    * ((F(values[t.control]) - F(t.target)) / F(t.scale)) ** 2
                    for t in model.objective.targets
                ),
                F(0),
            )
            if tracking
            else -output_value(model, "value", {**point, **values})
        )
        assert polynomial.value(system, values) == expected  # nosec B101


@pytest.mark.parametrize(
    "kind,name",
    (
        ("input", "load"),
        ("operating_limit", "cold"),
        ("operating_limit", "hot"),
        ("control_lower", "feed"),
        ("state_upper", "flow"),
    ),
)
def test_physical_change_rates_match_perturbed_declarations(kind, name):
    model, domain = plant()
    point, delta = {"load": 1}, F(1, 2)

    def bind(m):
        return m.as_claim(
            domain,
            requirements=("service",),
            operating=OperatingSpec.fixed(m.controls, {"feed": 2, "trim": 1}),
        )

    claim = bind(model)
    original = problem(claim, point)
    rates, constant = change_rates(
        claim, original, DispatchChange(kind, name, "MW", 0.5)
    )
    if kind == "input":
        point = {name: point[name] + float(delta)}
    elif kind == "operating_limit":
        model = replace(
            model,
            operating_limits=tuple(
                replace(r, limit=r.limit + float(delta)) if r.name == name else r
                for r in model.operating_limits
            ),
        )
    else:
        role, side = kind.split("_")
        field = "controls" if role == "control" else "states"
        model = replace(
            model,
            **{
                field: tuple(
                    replace(v, **{side: getattr(v, side) + float(delta)})
                    if v.name == name
                    else v
                    for v in getattr(model, field)
                )
            },
        )
    perturbed = problem(bind(model), point)
    assert original.rows == perturbed.rows and original.cost == perturbed.cost  # nosec B101
    assert [a + delta * h for a, h in zip(original.rhs, rates)] == perturbed.rhs  # nosec B101
    assert original.constant + delta * constant == perturbed.constant  # nosec B101
    assert all(
        rate == 0 for ref, rate in zip(original.refs, rates) if ref.startswith("fixed_")
    )  # nosec B101


def test_shared_scenario_relief_retains_physical_sources_and_scales():
    model, _ = plant()
    physical = compile_model(model, ("service",))
    originals = [physical.normalized({"load": load}, {}) for load in (0, 2)]
    stacked, blocks = stack_scenarios(iter(originals))
    amounts = {"hot": F(3, 2), "cold": F(1, 4)}
    augmented = augment_relief(
        stacked,
        (
            ReliefColumn(Variable(f"relief_{n}", "MW", 0, 4, "auxiliary"), n)
            for n in amounts
        ),
    )
    relaxed = relax_rows(stacked, amounts)
    dispatch = [F(1, 4), F(3, 4)] * 2
    coordinates = dispatch + [a / 4 for a in amounts.values()]
    for i, (row, rhs, source) in enumerate(
        zip(augmented.rows, augmented.upper, augmented.sources)
    ):
        original = originals[source.scenario]
        j = i % len(original.rows)
        assert source == replace(original.sources[j], scenario=source.scenario)  # nosec B101
        original_slack = original.upper[j] - sum(
            a * z for a, z in zip(original.rows[j], dispatch[:2])
        )
        relief = (
            amounts.get(source.name, F(0)) if source.kind == "operating_limit" else F(0)
        )
        expected = original_slack + relief / source.scale
        assert rhs - sum(a * z for a, z in zip(row, coordinates)) == expected  # nosec B101
        assert (
            relaxed.upper[i] - sum(a * z for a, z in zip(stacked.rows[i], dispatch))
            == expected
        )  # nosec B101
    for i, block in enumerate(blocks):
        assert block.offset == 2 * i  # nosec B101
        assert dict(block.mapping) == {
            v.source_name: v.name for v in stacked.variables if v.scenario == i
        }  # nosec B101
        assert tuple(v.role for v in stacked.variables if v.scenario == i) == (
            "control",
            "state",
        )  # nosec B101
    reduced = augmented.select_rows((len(originals[0].rows), 1))
    assert reduced.sources == (
        augmented.sources[len(originals[0].rows)],
        augmented.sources[1],
    )  # nosec B101
    assert reduced.row_refs == tuple(s.legacy_ref for s in reduced.sources)  # nosec B101
    for target in ("service", "control_lower:feed", "missing"):
        with pytest.raises(ValueError):
            augment_relief(
                stacked,
                (ReliefColumn(Variable("relief", "MW", 0, 4, "auxiliary"), target),),
            )
        with pytest.raises(ValueError):
            relax_rows(stacked, {target: 1})


def test_representation_rejects_misaligned_or_role_confused_problems():
    model, _ = plant()
    physical = compile_model(model, ("service",))
    with pytest.raises(ValueError, match="actuators"):
        physical.normalized({"load": 1}, {"flow": 2})
    with pytest.raises(ValueError, match="inputs"):
        physical.normalized({}, {})
    with pytest.raises(ValueError, match="duplicate"):
        replace(physical, variables=physical.variables * 2)
    with pytest.raises(ValueError, match="undeclared"):
        LinearProblem(
            (),
            (
                PhysicalRow(
                    AffineExpression(0, (("missing", 1),)),
                    0,
                    RowSource("requirement", "r", "MW"),
                ),
            ),
        )
    system = physical.normalized({"load": 1}, {})
    for changes in (
        {"rows": system.rows[:-1]},
        {"rows": [()] * len(system.rows)},
        {"sources": system.sources[:-1]},
        {"row_refs": ["wrong"] * len(system.rows)},
    ):
        with pytest.raises(ValueError, match="misaligned"):
            replace(system, **changes)
    with pytest.raises(ValueError):
        coordinates(((F(1),),), (F(2),), (F(0),), (F(0),), (F(1),))
    with pytest.raises(TypeError):
        system.fixed["trim"] = 2
    with pytest.raises(FrozenInstanceError):
        system.upper = ()


def test_box_objective_bounds_include_all_variables_without_explicit_rows():
    system = LinearSystem((Variable("x", "MW", -2, 6, "control"),), {}, (), (), ())
    assert system.objective_bound((F(-3),), ())[0] == -3  # nosec B101
    assert Polynomial(F(2), (F(-4),), (F(4),)).bound(system, ())[0] == 1  # nosec B101


def test_coordinate_scaling_remains_exact_for_integer_inputs():
    rows, rhs = coordinates(((1,),), (2,), (-1,), (2,), (3,))
    assert rows == ((F(2, 3),),) and rhs == (F(1),)  # nosec B101
