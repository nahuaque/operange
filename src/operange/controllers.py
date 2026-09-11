"""Immutable affine operating rules and their bounded linear-model binding."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import cached_property
import json
from typing import ClassVar

from .affine import AffineTerm
from ._numeric import exact_dot
from .claim import AdapterCapabilities, Capability, bind_contract, rejected_result
from .contract_types import Record, nonempty, reference, snapshot, unique
from .domains import FiniteSet
from .linear_process import LinearProcessAdapter
from .primitives import finite
from .recourse import DecisionRule, RecoursePolicy


def _read_json(document):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    return json.loads(document, object_pairs_hook=pairs, parse_constant=constant)


def _fields(data, expected):
    if not isinstance(data, Mapping) or set(data) != set(expected):
        raise ValueError("invalid controller artifact fields")


@dataclass(frozen=True)
class AffineControlRule(Record):
    """A physical command: offset + sum(coefficient * observed input).

    An empty term tuple declares a constant command. There is no implicit
    normalization, saturation, feedback state, fitting or retuning.
    """

    control: str
    unit: str
    offset: float
    terms: tuple[AffineTerm, ...] = ()
    stage: str = "operation"

    def _validate(self):
        for name in ("control", "unit", "stage"):
            nonempty(getattr(self, name), name)
        unique(tuple(t.variable for t in self.terms), "controller observations")

    @property
    def observes(self):
        return tuple(t.variable for t in self.terms)

    def command(self, observations):
        if not isinstance(observations, Mapping) or any(
            name not in observations for name in self.observes
        ):
            raise ValueError("controller observations are missing")
        values = tuple(finite(observations[n], n) for n in self.observes)
        exact = Fraction(self.offset) + exact_dot(
            (t.coefficient for t in self.terms), values
        )
        return finite(float(exact), self.control)


@dataclass(frozen=True)
class AffineController(Record):
    """A caller-supplied executable rule with a portable content identity."""

    name: str
    rules: tuple[AffineControlRule, ...]
    provenance: str
    schema_version: ClassVar[str] = "affine_controller/v1"

    def _validate(self):
        nonempty(self.name, "controller name")
        nonempty(self.provenance, "controller provenance")
        if not self.rules:
            raise ValueError("controller needs at least one rule")
        if any(type(r) is not AffineControlRule for r in self.rules):
            raise ValueError("portable controllers require declarative affine rules")
        unique(tuple(r.control for r in self.rules), "controller controls")

    def commands(self, observations):
        """Evaluate only the named observations, without clipping commands."""
        return {r.control: r.command(observations) for r in self.rules}

    @property
    def recourse_policy(self):
        return RecoursePolicy(
            "static" if any(r.terms for r in self.rules) else "fixed",
            tuple(
                DecisionRule(
                    r.control,
                    r.unit,
                    r.stage,
                    r.observes,
                    None if r.terms else r.offset,
                )
                for r in self.rules
            ),
        )

    def _body(self):
        return {"schema_version": self.schema_version, "controller": self.to_dict()}

    @cached_property
    def ref(self):
        return reference(self._body(), self.schema_version)

    @property
    def controller_id(self):
        return self.ref.artifact_id

    def to_manifest(self):
        return {**self._body(), "controller_id": self.controller_id}

    def to_json(self, *, indent=2):
        return json.dumps(
            self.to_manifest(), indent=indent, sort_keys=True, allow_nan=False
        )

    @classmethod
    def from_json(cls, document):
        data = _read_json(document)
        _fields(data, ("schema_version", "controller", "controller_id"))
        if data["schema_version"] != cls.schema_version:
            raise ValueError("unsupported affine controller schema")
        try:
            result = cls(**data["controller"])
        except TypeError as exc:
            raise ValueError("invalid affine controller fields") from exc
        if data["controller_id"] != result.controller_id:
            raise ValueError("controller identity mismatch")
        return result


@dataclass(frozen=True)
class _ControllerAdapter(Record):
    """Preserve the physical model and bind the controller as an operating rule."""

    model: LinearProcessAdapter
    controller: AffineController

    def _validate(self):
        if (
            type(self.model) is not LinearProcessAdapter
            or type(self.controller) is not AffineController
        ):
            raise ValueError(
                "controller binding supports the declarative linear adapter only"
            )

    @property
    def input_space(self):
        return self.model.input_space

    @cached_property
    def base_contract(self):
        original = self.model.base_contract
        operating = dict(original.operating)
        operating.pop("adjustable_observations")
        operating.update(
            {
                "response": "frozen_affine_controller",
                "controller": self.controller.to_manifest(),
                "feasibility_scope": "model_and_selected_requirements_under_frozen_controller",
                "objective": "execute prescribed commands; no optimization",
            }
        )
        return replace(
            original,
            operating=operating,
            numerical_policy={
                **dict(original.numerical_policy),
                "controller_arithmetic": "exact_rationals_of_declared_floats_then_round_command_to_float",
                "controller_clipping": False,
                "infeasibility": "verified violation by the uniquely prescribed commands; other controllers not assessed",
            },
        )

    def describe(self, claim):
        return bind_contract(
            self.base_contract, claim.domain, claim.recourse, claim.requirements
        )

    def validate_binding(self, policy):
        if not isinstance(policy, RecoursePolicy):
            raise ValueError("controller permissions must be a RecoursePolicy")
        if policy.mode not in ("static", "fixed"):
            raise ValueError(
                "affine controllers support a single static operation stage"
            )
        controls = {c.name: c for c in self.model.controls}
        if set(controls) != {r.control for r in self.controller.rules} or set(
            controls
        ) != {r.control for r in policy.rules}:
            raise ValueError(
                "controller and permissions need exactly one rule per model control"
            )
        for rule in self.controller.rules:
            permission = policy.rule(rule.control)
            if rule.unit != controls[rule.control].unit or permission.unit != rule.unit:
                raise ValueError(
                    "controller control units do not match the model and permissions"
                )
            if rule.stage != "operation" or permission.stage != "operation":
                raise ValueError(
                    "controller rules and permissions must use the operation stage"
                )
            if any(n not in self.input_space.names for n in permission.observes):
                raise ValueError(
                    "controller permissions reference unknown process observations"
                )
            if permission.fixed_value is not None:
                if rule.terms or rule.offset != permission.fixed_value:
                    raise ValueError(
                        "controller contradicts a fixed command permission"
                    )
            elif policy.mode == "fixed":
                raise ValueError("fixed permissions must fix every command")
            for term in rule.terms:
                if term.variable not in permission.observes:
                    raise ValueError(
                        "controller uses an observation outside its permissions"
                    )
                coordinate = self.input_space.coordinate(term.variable)
                if term.unit != f"{rule.unit}/{coordinate.unit}":
                    raise ValueError(
                        "controller coefficient units must match control/input units"
                    )

    def capabilities(self, claim):
        try:
            self.validate_binding(claim.recourse)
        except ValueError as exc:
            unavailable = Capability(False, str(exc))
            return AdapterCapabilities(*(unavailable for _ in range(5)))
        return AdapterCapabilities(
            Capability(
                True,
                "Direct execution and physical checks of the frozen controller at one domain member.",
            ),
            Capability(
                False,
                "Controller-response sensitivities are not provided in this slice.",
            ),
            Capability(
                type(claim.domain) is FiniteSet,
                "Complete finite-scenario replay of the frozen controller; no continuous-domain audit.",
            ),
            Capability(False, "Controller boundary searches are not provided."),
            Capability(
                False, "Controller breaking-distance searches are not provided."
            ),
        )

    def run(self, claim, operation, realization, options):
        from ._controller_results import audit_result, evaluate_result

        if options:
            return rejected_result(
                claim.contract,
                operation,
                snapshot({"query": operation, "realization": realization, **options}),
                "Unknown controller query options.",
                code="invalid_query_options",
                execution="invalid",
            )
        if operation == "evaluation":
            return evaluate_result(claim, realization)
        if operation == "audit":
            return audit_result(claim)
        return rejected_result(
            claim.contract,
            operation,
            {"query": operation},
            "Unsupported controller query.",
        )


def bind_controller(claim, controller):
    if type(claim.adapter) is LinearProcessAdapter:
        model = claim.adapter
    elif type(claim.adapter) is _ControllerAdapter:
        model = claim.adapter.model
    else:
        if type(controller) is AffineController:
            raise ValueError("affine controllers require LinearProcessAdapter")
        from .storage_controllers import bind_storage_controller

        return bind_storage_controller(claim, controller)
    adapter = _ControllerAdapter(model, controller)
    adapter.validate_binding(claim.recourse)
    return replace(claim, adapter=adapter)
