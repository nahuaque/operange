"""Explicit operating commitments built from actuator permissions and controllers.

Algebraic variables never appear in an operating specification. Adapters decide
which policy classes their studies can execute; declaring permissions alone does
not create a solver or a coverage guarantee.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from .recourse import DecisionRule, RecoursePolicy
from .primitives import finite


@dataclass(frozen=True)
class OperatingSpec:
    policy: RecoursePolicy
    controller: object | None = None

    def __post_init__(self):
        if not isinstance(self.policy, RecoursePolicy):
            raise ValueError("OperatingSpec needs a RecoursePolicy")
        if self.controller is not None:
            from .controllers import AffineController

            if type(self.controller) is not AffineController:
                raise ValueError(
                    "OperatingSpec supports declarative affine controllers"
                )

    @classmethod
    def fixed(cls, controls, commands):
        """Prescribe every physical actuator, leaving algebraic states to the model."""
        controls = tuple(controls)
        if not isinstance(commands, Mapping) or set(commands) != {
            c.name for c in controls
        }:
            raise ValueError("Fixed commands must cover exactly the physical actuators")
        return cls(
            RecoursePolicy(
                "fixed",
                tuple(
                    DecisionRule(
                        c.name,
                        c.unit,
                        "operation",
                        fixed_value=finite(commands[c.name], "actuator command"),
                    )
                    for c in controls
                ),
            )
        )

    @classmethod
    def dispatch(cls, controls, observations):
        """Permit independent dispatch after the named observations are available."""
        controls, observations = tuple(controls), tuple(observations)
        return cls(
            RecoursePolicy(
                "static" if controls else "fixed",
                tuple(
                    DecisionRule(c.name, c.unit, "operation", observations)
                    for c in controls
                ),
            )
        )

    @classmethod
    def shared(cls, controls):
        """Choose one common command vector, without observing scenario inputs."""
        return cls.dispatch(controls, ())

    @classmethod
    def controlled(cls, controller, *, permissions=None):
        """Execute a saved controller within explicit or controller-derived permissions."""
        return cls(
            controller.recourse_policy if permissions is None else permissions,
            controller,
        )

    def validate(self, controls, observations):
        controls = {c.name: c for c in controls}
        if {r.control for r in self.policy.rules} != set(controls):
            raise ValueError(
                "Operating permissions must cover exactly the physical actuators"
            )
        if self.policy.mode not in ("fixed", "static"):
            raise ValueError("OperatingSpec supports one fixed/static operation stage")
        for rule in self.policy.rules:
            if rule.unit != controls[rule.control].unit or rule.stage != "operation":
                raise ValueError(
                    "Actuator units/stage do not match the operating specification"
                )
            if not set(rule.observes) <= set(observations):
                raise ValueError("Operating permission references unknown observations")
            if self.policy.mode == "fixed" and rule.fixed_value is None:
                raise ValueError("Fixed permissions must fix every actuator")
        return self

    def point_commands(self, observations, *, measurements=False):
        """Resolve supported pointwise permissions; no partial-information shortcut."""
        if self.controller is not None:
            raise ValueError("A controller must be executed before fixing its commands")
        for rule in self.policy.rules:
            if rule.fixed_value is None:
                if measurements:
                    raise ValueError(
                        "measurement-based operation requires an explicit or synthesized controller"
                    )
                if self.policy.mode != "static" or set(rule.observes) != set(
                    observations
                ):
                    raise ValueError(
                        "each adjustable actuator must observe every input; shared or partial observations require a joint policy study"
                    )
        return {
            r.control: r.fixed_value
            for r in self.policy.rules
            if r.fixed_value is not None
        }

    def shared_commands(self):
        """Return prescribed entries of a policy whose remaining commands are shared."""
        if self.controller is not None or any(r.observes for r in self.policy.rules):
            raise ValueError(
                "Shared commands cannot observe inputs or execute a controller"
            )
        return {
            r.control: r.fixed_value
            for r in self.policy.rules
            if r.fixed_value is not None
        }

    def to_dict(self):
        return {
            "policy": self.policy.to_dict(),
            "controller": None
            if self.controller is None
            else self.controller.to_manifest(),
        }


def resolve_operating(
    controls, observations, *, operating=None, recourse=None, controller=None
):
    """Translate existing convenience arguments once, at an adapter boundary."""
    if operating is not None:
        if type(operating) is not OperatingSpec:
            raise ValueError("operating must be an OperatingSpec")
        if recourse is not None or controller is not None:
            raise ValueError("Supply operating or recourse/controller, not both")
        spec = operating
    elif controller is not None:
        spec = OperatingSpec.controlled(controller, permissions=recourse)
    elif recourse is not None:
        spec = OperatingSpec(recourse)
    else:
        spec = OperatingSpec.dispatch(controls, observations)
    return spec
