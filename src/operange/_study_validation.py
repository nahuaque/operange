"""Static request checks for the original adapter queries; no optimization."""

from collections.abc import Mapping

from .adapters import HeatRecoveryAdapter, ThermalStorageAdapter
from .affine import AffineProcessAdapter
from .controllers import _ControllerAdapter
from .domains import FiniteSet
from .linear_process import LinearProcessAdapter
from .objectives import ControlTrackingObjective
from .primitives import BoxSet, finite
from .storage_controllers import _StorageControllerAdapter


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _support_dependencies(domain):
    from .composition import Intersection, Product, Union

    if type(domain) is Intersection and domain.backend == "cvxpy":
        return {"cvxpy"}
    if type(domain) in (Intersection, Product, Union):
        return set().union(*(_support_dependencies(f) for f in domain.factors))
    return set()


def _sensitivity(claim, options):
    heat = type(claim.adapter) is HeatRecoveryAdapter
    response = "optimized" if heat else "fixed"
    if (
        options.get("response", response) != response
        or options.get("method", "analytical") != "analytical"
        or options.get("operator", "jacobian") not in ("jacobian", "directional")
        or options.get("perturbation_scope", "ambient") != "ambient"
    ):
        raise NotImplementedError(
            f"Only analytical ambient derivatives of the {response} response are supported."
        )
    if options.get("coordinate_space", "physical") not in (
        "physical",
        "normalized",
    ) or options.get("side", "two_sided") not in ("two_sided", "forward", "backward"):
        raise ValueError("invalid derivative coordinate space or side")
    if heat:
        from .heat_results import OUTPUTS

        known_outputs = OUTPUTS
    else:
        known_outputs = tuple(o.name for o in claim.adapter.outputs)
    inputs = options.get("inputs", claim.domain.space.names)
    outputs = options.get("outputs", known_outputs)
    for selected, known in (
        (inputs, claim.domain.space.names),
        (outputs, known_outputs),
    ):
        if (
            not isinstance(selected, (tuple, list))
            or not selected
            or any(n not in known for n in selected)
            or len(set(selected)) != len(selected)
        ):
            raise ValueError("derivative coordinates must be distinct declared names")
    direction = options.get("direction")
    if options.get("operator", "jacobian") == "directional":
        if not isinstance(direction, (tuple, list)) or len(direction) != len(inputs):
            raise ValueError("direction needs one finite value per input")
        for v in direction:
            finite(v, "direction")
    elif direction is not None:
        raise ValueError("a Jacobian cannot include a direction")
    if options.get("coordinate_space") == "normalized" and any(
        claim.domain.space.coordinate(n).nominal is None
        or claim.domain.space.coordinate(n).scale is None
        for n in inputs
    ):
        raise NotImplementedError(
            "Normalized derivatives require declared nominals and scales."
        )


def _distance(claim, operation, options):
    tolerance = finite(options.get("distance_tolerance", 1e-8), "distance_tolerance")
    if tolerance < 0:
        raise ValueError("distance_tolerance must be nonnegative")
    if type(claim.adapter) is HeatRecoveryAdapter:
        if (
            operation == "breaking"
            and finite(options.get("shortfall_margin_mw", 0.01), "shortfall_margin_mw")
            <= 10 * claim.adapter.model.tolerance
        ):
            raise ValueError(
                "shortfall_margin_mw must exceed ten times the numerical tolerance"
            )
        return
    from .distance import NormalizedLInf
    from .polytope import PolytopeSet

    if options.get("backend", "scipy") == "scipy" and (
        type(claim.domain) not in (BoxSet, PolytopeSet)
        or type(claim.distance) is not NormalizedLInf
    ):
        raise NotImplementedError(
            "Convex-domain and Euclidean distance searches require backend='cvxpy'."
        )
    controller = type(claim.adapter) is _ControllerAdapter
    rows = {
        r.constraint_id: r
        for r in claim.contract.constraints
        if r.constraint_id in claim.requirements
        or controller
        and r.kind == "operating_limit"
    }
    selected = options.get("constraints", tuple(rows))
    if (
        not isinstance(selected, (tuple, list))
        or not selected
        or any(n not in rows for n in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError(
            "constraints must select distinct included service or equipment constraints"
        )
    if operation == "breaking":
        margins = options.get("violation_margins")
        if not isinstance(margins, Mapping) or set(margins) != set(selected):
            raise ValueError(
                "violation_margins must give a physical-unit threshold for every selected constraint"
            )
        if any(
            finite(margins[n], "violation margin") <= rows[n].tolerance
            for n in selected
        ):
            raise ValueError(
                "each violation margin must be strictly greater than the constraint tolerance"
            )


def _linear(claim, operation, options):
    from ._domain_generators import generator_count
    from ._linear_diagnosis import validate_options

    model = claim.adapter
    backend = options.get("backend", "scipy")
    if isinstance(model.objective, ControlTrackingObjective) and backend != "cvxpy":
        raise NotImplementedError("Control tracking requires backend='cvxpy'.")
    relief = options.get("relief")
    if operation == "evaluation":
        validate_options(model, options.get("diagnose", False), relief)
    if operation == "audit" and type(claim.domain) is not FiniteSet:
        limit = options.get("max_vertices", 256)
        _positive_integer(limit, "max_vertices")
        count = generator_count(claim.domain)
        if count > limit:
            raise NotImplementedError(
                f"Complete coverage requires {count} generators, exceeding max_vertices={limit}; none were evaluated."
            )
    if operation == "audit" and relief is not None:
        from ._joint_relief import declarations

        if not isinstance(relief, Mapping) or "changes" not in relief:
            raise ValueError(
                "Shared relief needs changes and an explicit linear or quadratic objective."
            )
        declarations(model, {k: v for k, v in relief.items() if k != "max_scenarios"})
        limit = relief.get("max_scenarios", 256)
        _positive_integer(limit, "max_scenarios")
        count = generator_count(claim.domain)
        if count > limit:
            raise NotImplementedError(
                f"Shared relief needs all {count} scenarios/generators, exceeding max_scenarios={limit}; none were evaluated."
            )
    if (
        relief is not None
        and relief.get("objective") == "quadratic"
        and backend != "cvxpy"
    ):
        raise NotImplementedError("Quadratic relief requires backend='cvxpy'.")


def validate_basic(claim, operation, point, options):
    from ._study_dispatch import _backend
    from .integrations.idaes import IdaesAdapter
    from .pinch import HeatCascadeAdapter
    from .startup import StartupLoadAdapter

    adapter = claim.adapter
    kind = type(adapter)
    dependencies = _backend(options.get("backend", "scipy"))
    known = (
        LinearProcessAdapter,
        AffineProcessAdapter,
        _ControllerAdapter,
        HeatRecoveryAdapter,
        ThermalStorageAdapter,
        _StorageControllerAdapter,
        HeatCascadeAdapter,
        StartupLoadAdapter,
        IdaesAdapter,
    )
    if kind not in known:
        return "adapter_declared", (), dependencies
    if operation == "audit":
        dependencies = tuple(
            sorted(set(dependencies) | _support_dependencies(claim.domain))
        )
    joint = kind in (ThermalStorageAdapter, _StorageControllerAdapter)
    if operation == "evaluation":
        if joint:
            if point is not None:
                raise ValueError(
                    "This adapter evaluates the whole bound tree; a point query changes its scope."
                )
        elif kind is HeatRecoveryAdapter:
            adapter._model(claim).uncertainty.coordinates(point)
        else:
            if kind is IdaesAdapter and point is None:
                point = adapter.input_space.nominal
            claim.domain.space.validate(point)
    if operation == "sensitivity" and kind in (
        AffineProcessAdapter,
        HeatRecoveryAdapter,
    ):
        _sensitivity(claim, options)
        if kind is HeatRecoveryAdapter:
            adapter._model(claim).uncertainty.coordinates(point)
        else:
            claim.domain.space.validate(point)
    if operation in ("boundary", "breaking"):
        _distance(claim, operation, options)
    if kind is LinearProcessAdapter:
        _linear(claim, operation, options)
    if operation == "sensitivity":
        return "local", ("local_sensitivity",), dependencies
    if operation in ("boundary", "breaking"):
        return (
            "continuous_domain",
            ("distance_bounds", "continuous_coverage"),
            dependencies,
        )
    coverage = (
        "finite_tree"
        if joint
        else "point"
        if operation == "evaluation"
        else "finite_scenarios"
        if type(claim.domain) is FiniteSet
        else "continuous_domain"
    )
    evidence = set()
    if kind in known:
        evidence.add("checked_feasibility")
    if coverage == "continuous_domain":
        evidence.add("continuous_coverage")
    if kind is LinearProcessAdapter:
        evidence.update(("infeasibility_certificate", "independent_certificate"))
        if operation == "evaluation" and adapter.objective is not None:
            evidence.add("objective_bounds")
        if options.get("relief") is not None:
            evidence.add("relief_bounds")
    if kind is _ControllerAdapter:
        evidence.update(("controller_audit", "independent_certificate"))
    if kind is IdaesAdapter:
        dependencies = ("idaes", "pyomo")
    return coverage, evidence, dependencies
