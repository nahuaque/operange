"""Declarative causal rules for the two-period thermal-storage reference model."""

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from functools import cached_property
import json
from typing import ClassVar, Literal

from .claim import AdapterCapabilities, Capability, bind_contract, rejected_result
from .contract_types import Record, nonempty, reference, snapshot, unique
from .controllers import _fields, _read_json
from .domains import FiniteSet
from .primitives import finite
from .storage import ThermalStorageClaim
from .storage_types import (
    ScenarioTree,
    StorageDesign,
    StorageRequirement,
    StorageScenario,
)


@dataclass(frozen=True)
class StoragePreparationRule(Record):
    """A prescribed MW command for one observed preparation signal."""

    observation: str
    charge_mw: float

    def _validate(self):
        nonempty(self.observation, "preparation observation")


@dataclass(frozen=True)
class StorageController(Record):
    """Signal-indexed preparation followed by the revealed event's net heat load.

    Unlisted signals are unresolved unless a default command is declared. Event
    charging is max(recovered - demand, 0); discharge is max(demand - recovered,
    0). These are commands, without equipment or state saturation. The model's
    no-dumping, no-simultaneous-operation assumptions determine this event rule.
    """

    name: str
    preparation_rules: tuple[StoragePreparationRule, ...]
    provenance: str
    default_preparation_charge_mw: float | None = None
    event_rule: Literal["net_heat_balance"] = "net_heat_balance"
    schema_version: ClassVar[str] = "storage_controller/v1"

    def _validate(self):
        nonempty(self.name, "controller name")
        nonempty(self.provenance, "controller provenance")
        if any(type(r) is not StoragePreparationRule for r in self.preparation_rules):
            raise ValueError("portable storage controllers require declarative rules")
        unique(tuple(r.observation for r in self.preparation_rules), "signals")
        if not self.preparation_rules and self.default_preparation_charge_mw is None:
            raise ValueError("controller needs a preparation rule or explicit default")

    def preparation_command(self, observation):
        if not isinstance(observation, str) or not observation.strip():
            raise ValueError("preparation observation must be a nonempty string")
        for rule in self.preparation_rules:
            if rule.observation == observation:
                return rule.charge_mw
        if self.default_preparation_charge_mw is not None:
            return self.default_preparation_charge_mw
        raise ValueError(f"No preparation command for observation: {observation}")

    def event_commands(self, demand_mw, recovered_heat_mw):
        demand = finite(demand_mw, "demand_mw")
        recovered = finite(recovered_heat_mw, "recovered_heat_mw")
        if demand < 0 or recovered < 0:
            raise ValueError("event thermal powers must be nonnegative")
        surplus = Fraction(recovered) - Fraction(demand)
        return {
            "event_charge": finite(float(max(surplus, 0)), "event_charge"),
            "event_discharge": finite(float(max(-surplus, 0)), "event_discharge"),
        }

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
            raise ValueError("unsupported storage controller schema")
        try:
            result = cls(**data["controller"])
        except TypeError as exc:
            raise ValueError("invalid storage controller fields") from exc
        if data["controller_id"] != result.controller_id:
            raise ValueError("controller identity mismatch")
        return result


@dataclass(frozen=True)
class _StorageControllerAdapter:
    model: ThermalStorageClaim
    controller: StorageController

    def __post_init__(self):
        if (
            type(self.model) is not ThermalStorageClaim
            or type(self.controller) is not StorageController
        ):
            raise ValueError(
                "storage replay requires the declarative storage model and controller"
            )
        for value, expected in (
            (self.model.tree, ScenarioTree),
            (self.model.design, StorageDesign),
            (self.model.requirement, StorageRequirement),
            *((s, StorageScenario) for s in self.model.tree.scenarios),
        ):
            if type(value) is not expected:
                raise ValueError(
                    "storage replay requires declarative tree and physical settings"
                )

    @property
    def input_space(self):
        return self.model.tree.domain.space

    @cached_property
    def base_contract(self):
        from .storage_contract import contract_for

        original = contract_for(self.model)
        return replace(
            original,
            operating={
                **dict(original.operating),
                "response": "frozen_causal_storage_controller",
                "controller": self.controller.to_manifest(),
                "replay_tree": asdict(self.model.tree),
                "claim_name": self.model.name,
                "initial_state_rule": "declared initial energy for each independent two-period path",
                "state_transition": "carry prepared energy into event; no intermediate reset",
                "objective": "execute prescribed commands; no optimization",
                "feasibility_scope": "model_and_requirements_under_frozen_controller",
            },
            numerical_policy={
                **dict(original.numerical_policy),
                "controller_arithmetic": "exact rational arithmetic; round each command and stage state once to float",
                "residual_arithmetic": "exact residuals of carried float states and commands, rounded outward",
                "controller_clipping": False,
                "infeasibility": "violation by this controller; other controllers not assessed",
            },
        )

    def describe(self, claim):
        return bind_contract(
            self.base_contract, claim.domain, claim.recourse, claim.requirements
        )

    def validate_binding(self, policy):
        if policy != self.model.recourse_policy("causal"):
            raise ValueError(
                "storage replay requires the model's causal observation permissions"
            )

    def validate_claim(self, claim):
        self.validate_binding(claim.recourse)
        if (
            type(claim.domain) is not FiniteSet
            or claim.domain.ref != self.model.tree.domain.ref
        ):
            raise ValueError(
                "storage replay requires its bound tree; use as_claim(tree=...) for new paths"
            )
        expected = {
            c.constraint_id
            for c in self.base_contract.constraints
            if c.kind == "requirement"
        }
        if set(claim.requirements) != expected:
            raise ValueError(
                "storage replay requires all service and terminal requirements"
            )

    def capabilities(self, claim):
        try:
            self.validate_claim(claim)
        except ValueError as exc:
            return AdapterCapabilities(*(Capability(False, str(exc)) for _ in range(5)))
        yes = Capability(
            True,
            "Direct causal replay of every declared two-period path, in tree order.",
        )
        no = Capability(
            False, "Storage replay supports finite-tree evaluation and audit only."
        )
        return AdapterCapabilities(yes, no, yes, no, no)

    def run(self, claim, operation, realization, options):
        from ._storage_controller_results import audit_result, evaluate_result

        if options or realization is not None:
            return rejected_result(
                claim.contract,
                operation,
                snapshot({"query": operation, "realization": realization, **options}),
                "Replay evaluates the whole bound tree without query options; bind a new tree for held-out paths.",
                code="invalid_replay_request",
                execution="invalid",
            )
        return (
            evaluate_result(claim) if operation == "evaluation" else audit_result(claim)
        )


def bind_storage_controller(claim, controller):
    from .adapters import ThermalStorageAdapter

    if type(claim.adapter) not in (ThermalStorageAdapter, _StorageControllerAdapter):
        raise ValueError("storage controller requires a ThermalStorageClaim")
    adapter = _StorageControllerAdapter(claim.adapter.model, controller)
    adapter.validate_binding(claim.recourse)
    return replace(claim, adapter=adapter)


def restore_storage_claim(contract):
    """Reconstruct a known implementation without replaying or fitting rules."""
    from .contract_types import plain
    from .recourse import RecoursePolicy

    operating = plain(contract.operating)
    tree = operating["replay_tree"]
    _fields(tree, ("scenarios", "preparation_hours", "event_hours"))
    model = ThermalStorageClaim(
        operating["claim_name"],
        StorageDesign(**plain(contract.design)),
        ScenarioTree(
            tuple(StorageScenario(**s) for s in tree["scenarios"]),
            tree["preparation_hours"],
            tree["event_hours"],
        ),
        StorageRequirement(**plain(contract.model["requirement"])),
        contract.numerical_policy["absolute_tolerance"],
    )
    controller = StorageController.from_json(
        json.dumps(operating["controller"], allow_nan=False)
    )
    return replace(
        model.as_claim(controller=controller),
        recourse=RecoursePolicy(**operating["recourse_policy"]),
        requirements=operating["requirement_refs"],
    )
