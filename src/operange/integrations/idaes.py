"""Bind existing IDAES/Pyomo flowsheets without duplicating their equations.

The optional backend is loaded only when an adapter is constructed. Results
contain declarations and numerical evidence, never executable Pyomo objects.
"""

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from typing import Any, Literal

from ..claim import (
    AdapterCapabilities,
    Capability,
    Claim,
    bind_contract,
    rejected_result,
)
from ..contract_types import Record, freeze, nonempty, unique
from ..domains import FiniteSet, ParameterSpace
from ..recourse import DecisionRule, RecoursePolicy
from ._idaes_affine import IdaesLinearExport

__all__ = [
    "IdaesAdapter",
    "IdaesVariable",
    "IdaesRequirement",
    "IdaesInitialization",
    "IdaesLinearExport",
]


@dataclass(frozen=True)
class IdaesVariable(Record):
    """A scalar component identified by its Pyomo ComponentUID and native unit."""

    name: str
    component: str
    unit: str

    def _validate(self):
        for name in ("name", "component", "unit"):
            nonempty(getattr(self, name), name)


@dataclass(frozen=True)
class IdaesRequirement(Record):
    """A selectable, active, one-sided Pyomo constraint; tolerance in its unit."""

    name: str
    component: str
    unit: str
    tolerance: float = 1e-7

    def _validate(self):
        for name in ("name", "component", "unit"):
            nonempty(getattr(self, name), name)
        if self.tolerance < 0:
            raise ValueError("requirement tolerance must be nonnegative")


@dataclass(frozen=True)
class IdaesInitialization(Record):
    """Initialize a square unit using its upstream IDAES initialization API.

    ``block`` calls ``block.initialize(**options)``; ``default`` constructs
    ``block.default_initializer(**options)`` and calls its ``initialize``.
    Options are explicit JSON values, including any solver configuration.
    Each step runs on a disposable clone; only free variable guesses survive.
    """

    component: str
    method: Literal["block", "default"] = "block"
    options: Mapping[str, Any] = field(default_factory=dict)

    def _validate(self):
        nonempty(self.component, "initialization component")


@dataclass(frozen=True, kw_only=True)
class IdaesAdapter:
    """Snapshot a constructed continuous algebraic model, then solve clones.

    Inputs bind scalar Vars or mutable real-valued Params. Controls bind Vars;
    remaining unfixed variables are algebraic states. Requirements explicitly
    identify service constraints; all other active constraints and variable
    bounds always apply. Only fixed or fully observed static operation is
    supported. A numerical candidate never proves recourse infeasibility.
    """

    model: InitVar[object]
    name: str
    input_space: ParameterSpace
    inputs: tuple[IdaesVariable, ...]
    controls: tuple[IdaesVariable, ...]
    outputs: tuple[IdaesVariable, ...]
    requirements: tuple[IdaesRequirement, ...]
    solver: str = "highs"
    solver_options: Mapping[str, Any] = field(default_factory=dict)
    tolerance: float = 1e-7
    provenance: Mapping[str, Any] = field(default_factory=dict)
    initialization: tuple[IdaesInitialization, ...] = ()
    diagnostics: bool = False
    _template: object = field(init=False, repr=False, compare=False)
    _rows: tuple = field(init=False, repr=False)
    _states: tuple = field(init=False, repr=False)
    _contract: object = field(init=False, repr=False)

    def __post_init__(self, model):
        from ._idaes_model import build_contract, prepare_template

        nonempty(self.name, "model name")
        nonempty(self.solver, "solver name")
        if not isinstance(self.input_space, ParameterSpace):
            raise ValueError("input_space must be a ParameterSpace")
        if not 0 <= self.tolerance <= 1e-3:
            raise ValueError("tolerance must lie between zero and 1e-3")
        if type(self.diagnostics) is not bool:
            raise ValueError("diagnostics must be a bool")
        for name, cls in (
            ("inputs", IdaesVariable),
            ("controls", IdaesVariable),
            ("outputs", IdaesVariable),
            ("requirements", IdaesRequirement),
            ("initialization", IdaesInitialization),
        ):
            items = tuple(getattr(self, name))
            if any(type(item) is not cls for item in items):
                raise ValueError(f"{name} need {cls.__name__} declarations")
            if name != "initialization":
                unique(tuple(item.name for item in items), name)
            unique(tuple(item.component for item in items), f"{name} components")
            object.__setattr__(self, name, items)
        if not self.outputs or not self.requirements:
            raise ValueError("IDAES bindings need outputs and service requirements")
        unique(
            tuple(v.name for v in self.inputs + self.controls + self.outputs),
            "quantity names",
        )
        unique(
            tuple(v.component for v in self.inputs + self.controls),
            "input/control components",
        )
        if {v.name for v in self.inputs} != set(self.input_space.names):
            raise ValueError("input bindings must cover exactly the input space")
        for item in self.inputs:
            if item.unit != self.input_space.coordinate(item.name).unit:
                raise ValueError("input binding units must match the input space")
        for name in ("solver_options", "provenance"):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError(f"{name} must be a mapping")
            object.__setattr__(self, name, freeze(getattr(self, name)))
        template, rows, states = prepare_template(self, model)
        object.__setattr__(self, "_template", template)
        object.__setattr__(self, "_rows", rows)
        object.__setattr__(self, "_states", states)
        object.__setattr__(self, "_contract", build_contract(self))

    @property
    def base_contract(self):
        return self._contract

    def describe(self, claim):
        return bind_contract(
            self.base_contract, claim.domain, claim.recourse, claim.requirements
        )

    def _permissions(self, policy):
        if policy.mode not in ("fixed", "static"):
            raise ValueError(
                "IDAES supports fixed or fully observed static operation only"
            )
        if {r.control for r in policy.rules} != {v.name for v in self.controls}:
            raise ValueError("one operating permission is required for each actuator")
        for rule in policy.rules:
            item = next(v for v in self.controls if v.name == rule.control)
            if rule.unit != item.unit or rule.stage != "operation":
                raise ValueError("actuator units/stage do not match the binding")
            if rule.fixed_value is None and (
                policy.mode != "static"
                or set(rule.observes) != set(self.input_space.names)
            ):
                raise ValueError("adjustable actuators must observe every input")

    def as_claim(self, domain, *, recourse=None, requirements=None):
        if recourse is None:
            recourse = RecoursePolicy(
                "static" if self.controls else "fixed",
                tuple(
                    DecisionRule(v.name, v.unit, "operation", self.input_space.names)
                    for v in self.controls
                ),
            )
        return Claim(self, domain, recourse, requirements)

    def capabilities(self, claim):
        try:
            self._permissions(claim.recourse)
        except ValueError as exc:
            return AdapterCapabilities(*(Capability(False, str(exc)) for _ in range(5)))
        unsupported = Capability(
            False,
            "No nonlinear sensitivity, distance or continuous-domain proof is supplied by this integration.",
        )
        return AdapterCapabilities(
            Capability(
                True,
                "Numerical candidate with all active residuals and bounds checked; no infeasibility or optimality certificate.",
            ),
            unsupported,
            Capability(
                type(claim.domain) is FiniteSet,
                "Complete declared finite scenarios only; unresolved solves remain inconclusive.",
            ),
            unsupported,
            unsupported,
        )

    def run(self, claim, operation, realization, options):
        from ._idaes_results import audit_result, evaluate_result

        if options:
            return rejected_result(
                claim.contract,
                operation,
                {"query": operation, "realization": realization, **options},
                "Solver settings belong to the adapter binding; query options are unsupported.",
                code="invalid_query_options",
                execution="invalid",
            )
        if operation == "evaluation":
            return evaluate_result(claim, realization)
        return audit_result(claim)

    def export_linear(self):
        """Export supported, exactly representable affine arithmetic; never linearize."""
        from ._idaes_affine import export_linear

        return export_linear(self)
