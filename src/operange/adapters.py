"""Bindings from the two reference models to the common process DSL."""

from dataclasses import dataclass, replace

from .claim import AdapterCapabilities, Capability, bind_contract, rejected_result
from .distance import NormalizedLInf
from .domains import FiniteSet
from .heat_recovery import HeatRecoveryClaim
from .primitives import BoxSet
from .storage import ThermalStorageClaim


def _unsupported(reason):
    capability = Capability(False, reason)
    return AdapterCapabilities(
        capability, capability, capability, capability, capability
    )


def _requirements(contract):
    return tuple(
        sorted(c.constraint_id for c in contract.constraints if c.kind == "requirement")
    )


def _invalid_options(claim, operation, realization, options, allowed):
    if set(options) - set(allowed):
        return rejected_result(
            claim.contract,
            operation,
            {"query": operation, "realization": realization, **options},
            "Unknown options for this adapter query: "
            + ", ".join(sorted(set(options) - set(allowed))),
            code="invalid_query_options",
            execution="invalid",
        )
    return None


@dataclass(frozen=True)
class HeatRecoveryAdapter:
    model: HeatRecoveryClaim

    def __post_init__(self):
        if not isinstance(self.model, HeatRecoveryClaim):
            raise ValueError("HeatRecoveryAdapter needs a HeatRecoveryClaim model")

    @property
    def base_contract(self):
        from .heat_results import contract_for

        return contract_for(self.model)

    @property
    def input_space(self):
        return self.model.uncertainty.space

    def _model(self, claim):
        if type(claim.domain) is not BoxSet:
            raise ValueError(
                "This heat adapter supports a BoxSet only; its monotonicity proof cannot be transferred to another domain."
            )
        return replace(self.model, uncertainty=claim.domain)

    def describe(self, claim):
        from .heat_results import contract_for

        try:
            model = self._model(claim)
        except ValueError:
            model = self.model
        return bind_contract(
            contract_for(model), claim.domain, claim.recourse, claim.requirements
        )

    def capabilities(self, claim):
        try:
            model = self._model(claim)
        except ValueError as exc:
            return _unsupported(str(exc))
        if claim.recourse != model.recourse_policy:
            return _unsupported(
                "This heat adapter supports fully observed static electrical-power recourse only."
            )
        if claim.requirements != _requirements(self.base_contract):
            return _unsupported(
                "This heat adapter requires its declared heat-demand requirement."
            )
        distance_ok = type(claim.distance) is NormalizedLInf and all(
            claim.distance.space.coordinate(p.name).nominal == p.nominal
            and claim.distance.space.coordinate(p.name).scale == p.scale
            for p in model.uncertainty.parameters
        )
        search = Capability(
            distance_ok,
            "Exhaustive affine branches under the box's declared normalized L-infinity scales and baseline."
            if distance_ok
            else "Boundary and breaking searches require NormalizedLInf with the box's declared baseline and scales.",
        )
        return AdapterCapabilities(
            Capability(
                True, "Optimized constant-COP static response at one box member."
            ),
            Capability(
                True,
                "Analytical optimized-response Jacobian/directional derivatives only.",
            ),
            Capability(
                True, "Full box coverage by the model-specific monotonicity proof."
            ),
            search,
            search,
        )

    def run(self, claim, operation, realization, options):
        from .heat_results import audit_result, evaluate_result, threshold_result
        from .heat_sensitivity import sensitivity_result

        allowed = (
            (
                "inputs",
                "outputs",
                "operator",
                "coordinate_space",
                "response",
                "side",
                "direction",
                "method",
            )
            if operation == "sensitivity"
            else ("shortfall_margin_mw",)
            if operation == "breaking"
            else ()
        )
        invalid = _invalid_options(claim, operation, realization, options, allowed)
        if invalid:
            return invalid
        model = self._model(claim)
        if operation == "evaluation":
            return evaluate_result(model, realization)
        if operation == "sensitivity":
            return sensitivity_result(model, realization, **options)
        if operation == "audit":
            return audit_result(model)
        return threshold_result(model, boundary=operation == "boundary", **options)


@dataclass(frozen=True)
class ThermalStorageAdapter:
    model: ThermalStorageClaim

    def __post_init__(self):
        if not isinstance(self.model, ThermalStorageClaim):
            raise ValueError("ThermalStorageAdapter needs a ThermalStorageClaim model")

    @property
    def base_contract(self):
        from .storage_contract import contract_for

        return contract_for(self.model)

    @property
    def input_space(self):
        return self.model.tree.domain.space

    def _operating(self, claim):
        mode = claim.recourse.mode
        fixed = claim.recourse.rule("preparation_charge").fixed_value
        expected = self.model.recourse_policy(mode, fixed_preparation_power_mw=fixed)
        if claim.recourse != expected:
            raise ValueError(
                "The storage adapter requires its declared preparation/event observation permissions and controls."
            )
        return mode, fixed

    def describe(self, claim):
        from .storage_contract import contract_for

        try:
            mode, fixed = self._operating(claim)
        except ValueError:
            mode, fixed = "causal", None
        return bind_contract(
            contract_for(self.model, mode, fixed),
            claim.domain,
            claim.recourse,
            claim.requirements,
        )

    def capabilities(self, claim):
        if (
            type(claim.domain) is not FiniteSet
            or claim.domain.ref != self.model.tree.domain.ref
        ):
            return _unsupported(
                "This storage adapter supports the finite futures bound to its two-period model. A different domain requires a corresponding model/tree binding."
            )
        try:
            self._operating(claim)
        except ValueError as exc:
            return _unsupported(str(exc))
        if claim.requirements != _requirements(self.base_contract):
            return _unsupported(
                "The storage adapter evaluates all declared service and terminal requirements jointly."
            )
        derivative = Capability(
            False,
            "The storage adapter does not provide derivatives of its finite-tree policy solution.",
        )
        search = Capability(
            False,
            "The storage adapter provides conflicting-future cardinality, not a continuous-distance search.",
        )
        return AdapterCapabilities(
            Capability(
                True,
                "Joint model-and-requirements feasibility on the complete declared tree.",
            ),
            derivative,
            Capability(
                True,
                "Finite-tree policy feasibility, with an independent interval certificate.",
            ),
            search,
            search,
        )

    def run(self, claim, operation, realization, options):
        from .storage_contract import audit_result, evaluate_result

        invalid = _invalid_options(claim, operation, realization, options, ())
        if invalid:
            return invalid
        if realization is not None:
            return rejected_result(
                claim.contract,
                operation,
                {"query": operation, "realization": realization},
                "Storage evaluation is joint over the declared tree; a point query would change the claim's operating scope.",
                code="invalid_evaluation_scope",
                execution="invalid",
            )
        mode, fixed = self._operating(claim)
        if operation == "evaluation":
            return evaluate_result(self.model, mode, fixed_preparation_power_mw=fixed)
        return audit_result(self.model, mode, fixed_preparation_power_mw=fixed)
