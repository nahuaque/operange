"""Portable replay of supported model/controller engineering bindings."""

from dataclasses import dataclass, replace
from functools import cached_property
import json
from typing import ClassVar

from .claim import Claim
from .contract_types import ProcessContract, plain, reference
from .controllers import AffineController, _ControllerAdapter, _fields, _read_json
from .distance import NormalizedLInf
from .domain_io import domain_from_manifest
from .domains import ParameterSpace
from .linear_process import LinearProcessAdapter
from .recourse import RecoursePolicy


def _same_domain_implementations(original, restored):
    if type(original) is not type(restored):
        return False
    # Compositions may contain consumer-defined factors with built-in-looking
    # manifests. Check nested implementations before promising faithful replay.
    first, second = getattr(original, "factors", ()), getattr(restored, "factors", ())
    if len(first) != len(second) or any(
        not _same_domain_implementations(a, b) for a, b in zip(first, second)
    ):
        return False
    if hasattr(original, "envelope"):
        return _same_domain_implementations(original.envelope, restored.envelope)
    return True


@dataclass(frozen=True)
class FrozenController:
    """An immutable executable controller, physical model, domain and permissions.

    Freezing does not require a passing audit: failed controllers remain useful
    reproducible counterexamples. Identities establish content, not authorship.
    Loading reconstructs only the supported declarative implementation, without
    evaluating commands or invoking a model, optimizer or arbitrary callback.
    """

    claim: Claim
    schema_version: ClassVar[str] = "frozen_controller/v1"

    def __post_init__(self):
        if not isinstance(self.claim, Claim):
            raise ValueError("freeze requires a claim with an explicit controller")
        if type(self.claim.adapter) is not _ControllerAdapter:
            from .storage_controllers import _StorageControllerAdapter

            if type(self.claim.adapter) is not _StorageControllerAdapter:
                raise ValueError(
                    "freeze requires an explicit supported controller binding"
                )
            self.claim.adapter.validate_claim(self.claim)
        self.claim.adapter.validate_binding(self.claim.recourse)
        # Refuse to promise portable replay of consumer-defined domain behavior.
        domain = domain_from_manifest(self.claim.domain.to_manifest())
        if (
            not _same_domain_implementations(self.claim.domain, domain)
            or domain.ref != self.claim.domain.ref
        ):
            raise ValueError("frozen replay requires a supported declarative domain")
        if (
            self.claim.distance is not None
            and type(self.claim.distance) is not NormalizedLInf
        ):
            raise ValueError(
                "frozen replay can preserve NormalizedLInf distance declarations only"
            )
        self.claim.contract

    @property
    def controller(self):
        return self.claim.adapter.controller

    @property
    def model(self):
        return self.claim.adapter.model

    @property
    def contract(self):
        return self.claim.contract

    def as_claim(self, domain=None, *, tree=None):
        """Replay a new domain explicitly; the original frozen binding is retained."""
        if tree is not None:
            from .storage_controllers import _StorageControllerAdapter
            from .storage_types import ScenarioTree

            if (
                domain is not None
                or type(self.claim.adapter) is not _StorageControllerAdapter
            ):
                raise ValueError(
                    "tree rebinding requires a frozen storage controller and no domain argument"
                )
            if type(tree) is not ScenarioTree:
                raise ValueError("tree must be a declarative ScenarioTree")
            model = replace(self.model, tree=tree)
            return replace(
                self.claim,
                adapter=_StorageControllerAdapter(model, self.controller),
                domain=tree.domain,
            )
        return self.claim if domain is None else replace(self.claim, domain=domain)

    def evaluate_result(self, realization=None, **options):
        return self.claim.evaluate_result(realization, **options)

    def audit_result(self, **options):
        return self.claim.audit_result(**options)

    def _body(self):
        return {
            "schema_version": self.schema_version,
            "contract": self.contract.to_dict(),
            "distance": None
            if self.claim.distance is None
            else {
                "kind": "normalized_linf",
                "space": self.claim.distance.space.to_dict(),
            },
        }

    @cached_property
    def ref(self):
        return reference(self._body(), self.schema_version)

    @property
    def frozen_id(self):
        return self.ref.artifact_id

    @property
    def fingerprint(self):
        return self.ref.digest

    def to_dict(self):
        return {**self._body(), "frozen_id": self.frozen_id}

    def to_json(self, *, indent=2):
        return json.dumps(
            self.to_dict(), indent=indent, sort_keys=True, allow_nan=False
        )

    @classmethod
    def from_json(cls, document):
        data = _read_json(document)
        _fields(data, ("schema_version", "contract", "distance", "frozen_id"))
        if data["schema_version"] != cls.schema_version:
            raise ValueError("unsupported frozen controller schema")
        try:
            contract = ProcessContract(**data["contract"])
            if contract.model_id == "linear_process/v1":
                model = LinearProcessAdapter(**plain(contract.model))
                controller = AffineController.from_json(
                    json.dumps(plain(contract.operating["controller"]), allow_nan=False)
                )
                domain = domain_from_manifest(plain(contract.domain))
                recourse = RecoursePolicy(
                    **plain(contract.operating["recourse_policy"])
                )
                claim = model.as_claim(
                    domain,
                    recourse=recourse,
                    controller=controller,
                    requirements=contract.operating["requirement_refs"],
                )
            elif contract.model_id == "thermal_storage_two_period_v1":
                from .storage_controllers import restore_storage_claim

                claim = restore_storage_claim(contract)
            else:
                raise ValueError("unsupported frozen controller model")
            if data["distance"] is not None:
                distance = data["distance"]
                _fields(distance, ("kind", "space"))
                if distance["kind"] != "normalized_linf":
                    raise ValueError("unsupported frozen distance declaration")
                claim = replace(
                    claim, distance=NormalizedLInf(ParameterSpace(**distance["space"]))
                )
            if claim.contract.ref != contract.ref:
                raise ValueError(
                    "frozen contract does not match its executable model and controller"
                )
            result = cls(claim)
            if result.frozen_id != data["frozen_id"]:
                raise ValueError("frozen controller identity mismatch")
            return result
        except (TypeError, KeyError, AttributeError) as exc:
            raise ValueError("invalid frozen controller fields") from exc
