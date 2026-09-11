"""Finite-scenario startup audits of supplied scalar load profiles.

Scheduled start commands are fixed controls; amplitude, duration scale and
timing jitter are uncertain inputs. No motor equations or controller synthesis
are inferred. Exact rational arithmetic covers every linear time segment.
"""

from dataclasses import dataclass

from .claim import (
    AdapterCapabilities,
    Capability,
    Claim,
    bind_contract,
    rejected_result,
)
from .contract_types import (
    ConstraintSpec,
    ProcessContract,
    QuantitySpec,
    Record,
    nonempty,
    snapshot,
    unique,
)
from .domains import Coordinate, FiniteSet, ParameterSpace
from .profiles import PiecewiseLinearProfile
from .recourse import DecisionRule, RecoursePolicy


@dataclass(frozen=True)
class StartupEvent(Record):
    """One off -> starting -> running transition with a fixed command time.

    The profile's last value is the running load, held through the horizon.
    Amplitude uncertainty scales the entire profile, including its running tail.
    """

    name: str
    profile: PiecewiseLinearProfile
    scheduled_start_seconds: float
    provenance: str

    def _validate(self):
        nonempty(self.name, "event name")
        nonempty(self.provenance, "event provenance")
        if self.scheduled_start_seconds < 0 or any(v < 0 for v in self.profile.values):
            raise ValueError("startup commands and load profiles must be nonnegative")

    def input_name(self, field):
        if field not in ("amplitude_scale", "duration_scale", "timing_jitter_seconds"):
            raise ValueError("unknown startup input")
        return f"{self.name}:{field}"

    @property
    def control_name(self):
        return f"{self.name}:scheduled_start"


@dataclass(frozen=True)
class StartupLoadAdapter(Record):
    """Add supplied load envelopes and audit a complete finite uncertainty set.

    All events start and finish starting within [0, horizon_seconds]. Profiles
    share one scalar load unit. Summation, constant background load and held
    running tails are explicit model assumptions, not an electrical network
    calculation. Integral limits use load-unit * seconds, not inferred energy
    or a motor-protection thermal model.
    """

    name: str
    events: tuple[StartupEvent, ...]
    horizon_seconds: float
    capacity: float
    background_load: float = 0.0
    integral_limit: float | None = None
    tolerance: float = 1e-8
    integral_tolerance: float = 1e-8

    def _validate(self):
        nonempty(self.name, "model name")
        if not self.events:
            raise ValueError("startup model needs events")
        unique(tuple(e.name for e in self.events), "startup events")
        if len({e.profile.unit for e in self.events}) != 1:
            raise ValueError("startup profiles must share one load unit")
        if (
            self.horizon_seconds <= 0
            or self.capacity < 0
            or self.background_load < 0
            or self.tolerance < 0
            or self.integral_tolerance < 0
            or (self.integral_limit is not None and self.integral_limit < 0)
        ):
            raise ValueError("invalid startup horizon, load limit or tolerance")
        self.base_contract

    @property
    def unit(self):
        return self.events[0].profile.unit

    @property
    def input_space(self):
        return ParameterSpace(
            tuple(
                Coordinate(e.input_name(field), unit, e.provenance, nominal, 1)
                for e in self.events
                for field, unit, nominal in (
                    ("amplitude_scale", "1", 1),
                    ("duration_scale", "1", 1),
                    ("timing_jitter_seconds", "s", 0),
                )
            )
        )

    @property
    def recourse_policy(self):
        return RecoursePolicy(
            "fixed",
            tuple(
                DecisionRule(
                    e.control_name,
                    "s",
                    "before_startup",
                    fixed_value=e.scheduled_start_seconds,
                )
                for e in self.events
            ),
        )

    @property
    def base_contract(self):
        quantities = [
            QuantitySpec(c.name, c.unit, "startup_uncertainty", "input")
            for c in self.input_space.coordinates
        ]
        quantities.extend(
            QuantitySpec(e.control_name, "s", "command_time", "control")
            for e in self.events
        )
        quantities.extend(
            (
                QuantitySpec("peak_load", self.unit, "scalar_load_envelope", "output"),
                QuantitySpec("peak_time", "s", "elapsed_time", "output"),
                QuantitySpec(
                    "integrated_load", f"{self.unit}*s", "load_time_integral", "output"
                ),
                QuantitySpec(
                    "trajectory_residual", self.unit, "profile_definition", "residual"
                ),
                QuantitySpec(
                    "capacity_residual", self.unit, "capacity_margin", "residual"
                ),
            )
        )
        constraints = [
            ConstraintSpec(
                "trajectory_definition",
                "trajectory_residual",
                "equation",
                "eq",
                "load = background + sum(transformed supplied profiles)",
                0,
            ),
            ConstraintSpec(
                "shared_capacity",
                "capacity_residual",
                "requirement",
                "le",
                "maximum load over the entire horizon - capacity",
                self.tolerance,
            ),
        ]
        if self.integral_limit is not None:
            quantities.append(
                QuantitySpec(
                    "integral_residual", f"{self.unit}*s", "integral_margin", "residual"
                )
            )
            constraints.append(
                ConstraintSpec(
                    "startup_exposure",
                    "integral_residual",
                    "requirement",
                    "le",
                    "integral of load over the horizon - integral limit",
                    self.integral_tolerance,
                )
            )
        return ProcessContract(
            self.name,
            {
                "kind": "supplied_startup_profiles",
                "events": [
                    {
                        "name": e.name,
                        "profile": e.profile.to_dict(),
                        "provenance": e.provenance,
                    }
                    for e in self.events
                ],
                "interpolation": "piecewise_linear",
                "before_start": "off_zero_load",
                "after_startup": "hold_final_running_load",
                "amplitude_scaling": "entire_profile_including_running_tail",
                "load_combination": "additive_scalar_envelopes",
            },
            {
                "capacity": self.capacity,
                "unit": self.unit,
                "background_load": self.background_load,
                "integral_limit": self.integral_limit,
            },
            {},
            {
                "horizon_seconds": self.horizon_seconds,
                "initial_mode": "off_before_each_start",
                "schedule": {e.name: e.scheduled_start_seconds for e in self.events},
            },
            tuple(quantities),
            tuple(constraints),
            {
                "absolute_tolerance": self.tolerance,
                "integral_tolerance": self.integral_tolerance,
                "arithmetic": "exact_rationals_of_declared_floats",
                "residual_export": "rounded_toward_positive_infinity",
            },
        )

    def describe(self, claim):
        return bind_contract(
            self.base_contract, claim.domain, claim.recourse, claim.requirements
        )

    def capabilities(self, claim):
        if claim.recourse != self.recourse_policy:
            no = Capability(
                False,
                "Startup audit requires the declared fixed start schedule; adaptive scheduling is unsupported.",
            )
            return AdapterCapabilities(*(no for _ in range(5)))
        if type(claim.domain) is not FiniteSet:
            no = Capability(
                False,
                "This startup adapter covers an explicitly finite set of profile-parameter realizations only.",
            )
            return AdapterCapabilities(*(no for _ in range(5)))
        no = Capability(
            False,
            "Startup sensitivity and nearest-failure searches are not implemented.",
        )
        return AdapterCapabilities(
            Capability(
                True, "Piecewise-linear profiles over the full declared time horizon."
            ),
            no,
            Capability(
                True,
                "Complete finite uncertainty enumeration and all time segments of each declared profile.",
            ),
            no,
            no,
        )

    def as_claim(self, domain, *, requirements=None, recourse=None):
        return Claim(
            self,
            domain,
            self.recourse_policy if recourse is None else recourse,
            requirements,
        )

    def run(self, claim, operation, realization, options):
        from ._startup_results import _audit, _evaluate

        if options:
            return rejected_result(
                claim.contract,
                operation,
                {"realization": snapshot(realization), **snapshot(options)},
                "Unknown startup query options.",
                execution="invalid",
                code="invalid_options",
            )
        if operation == "evaluation":
            return _evaluate(claim, realization)
        if operation == "audit":
            return _audit(claim)
        return rejected_result(
            claim.contract, operation, {}, "Unsupported startup operation."
        )
