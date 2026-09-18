"""Bind existing IDAES/Pyomo flowsheets without duplicating their equations.

The optional backend is loaded only when an adapter is constructed. Results
contain declarations and numerical evidence, never executable Pyomo objects.
"""

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field, replace
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
from ..operating import OperatingSpec, resolve_operating
from ._idaes_affine import IdaesLinearExport
from ._idaes_shared_result import IdaesSharedControlResult
from ._idaes_uncertainty_result import (
    IdaesUncertaintyResult,
    IdaesUncertainParameter,
    IdaesResponse,
    IdaesPerturbation,
)

__all__ = [
    "IdaesAdapter",
    "IdaesVariable",
    "IdaesRequirement",
    "IdaesInitialization",
    "IdaesLinearExport",
    "IdaesSharedControlResult",
    "IdaesUncertaintyResult",
    "IdaesUncertainParameter",
    "IdaesResponse",
    "IdaesPerturbation",
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
        return (
            OperatingSpec(policy)
            .validate(self.controls, self.input_space.names)
            .point_commands(self.input_space.names)
        )

    def as_claim(self, domain, *, recourse=None, operating=None, requirements=None):
        spec = resolve_operating(
            self.controls,
            self.input_space.names,
            operating=operating,
            recourse=recourse,
        )
        if spec.controller is not None:
            raise ValueError(
                "IDAES controller execution is not supported; export a supported linear model first"
            )
        return Claim(self, domain, spec.policy, requirements)

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

    def optimize_shared_controls(self, domain, *, operating=None):
        """Optimize one command vector over a FiniteSet of input realizations.

        Uses the source model's active objective and sense, aggregated as the
        worst case. All active requirements apply in every scenario. A returned
        command is checked against the original equations at every declared
        point; nonlinear global optimality and infeasibility are not certified.
        """
        from ._idaes_shared_controls import optimize_shared_controls

        return optimize_shared_controls(self, domain, operating=operating)

    def with_parameters(self, point, *, parameter_map):
        """Bind provider-independent parameter values on a fresh model snapshot."""
        from ..contract_types import plain
        from ._idaes_parameters import parameter_model

        model, assignments = parameter_model(self, point, parameter_map)
        return replace(
            self,
            model=model,
            provenance={
                **plain(self.provenance),
                "parameters": {
                    "point": point.to_dict(),
                    "parameter_map": dict(parameter_map),
                    "parent_contract_ref": self.base_contract.ref.to_dict(),
                    "updates": [plain(a) for a in assignments],
                },
            },
        )

    def propagate_uncertainty(
        self,
        *,
        parameters,
        covariance,
        realization=None,
        outputs=None,
        margins=(),
        fixed_controls=None,
        reoptimize=False,
        operating=None,
        steps=None,
        relative_tolerance=1e-3,
        absolute_tolerance=1e-6,
    ):
        """Local output/margin covariance in explicit parameter and response order.

        Parameters bind fixed Vars or mutable Params outside the input/control
        bindings. Their current values define the linearization point. Supply
        all fixed commands, or opt into ideal reoptimization with explicit
        native-unit finite-difference steps. Fixed commands use Pyomo automatic
        differentiation of a regular, square equality-state model. Reoptimized
        responses require checked symmetric solves at steps h and h/2.
        """
        from ._idaes_uncertainty import propagate_uncertainty

        return propagate_uncertainty(
            self,
            parameters=parameters,
            covariance=covariance,
            realization=realization,
            outputs=outputs,
            margins=margins,
            fixed_controls=fixed_controls,
            reoptimize=reoptimize,
            operating=operating,
            steps=steps,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
