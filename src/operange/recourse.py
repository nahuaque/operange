"""Serializable permissions for decision rules; no executable controllers."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .contract_types import Record, freeze, nonempty, reference, unique


@dataclass(frozen=True)
class DecisionRule(Record):
    control: str
    unit: str
    stage: str
    observes: tuple[str, ...] = ()
    fixed_value: float | None = None

    def _validate(self):
        for name in ("control", "unit", "stage"):
            nonempty(getattr(self, name), name)
        unique(self.observes, "observation names")
        for name in self.observes:
            nonempty(name, "observation name")
        if self.fixed_value is not None and self.observes:
            raise ValueError("fixed controls cannot depend on observations")

    def information(self, observations):
        """Return only information this rule may use; future fields are excluded."""
        if not isinstance(observations, Mapping) or any(
            name not in observations for name in self.observes
        ):
            raise ValueError("required observations are missing")
        return freeze({name: observations[name] for name in self.observes})


@dataclass(frozen=True)
class RecoursePolicy(Record):
    """A permitted policy class or fixed decisions, rather than a fitted policy.

    Stage ordering and observation availability are validated by the adapter.
    A fixed preparation decision can coexist with adjustable event decisions.
    """

    mode: Literal["static", "fixed", "causal", "perfect_foresight"]
    rules: tuple[DecisionRule, ...]

    def _validate(self):
        if not self.rules and self.mode != "fixed":
            raise ValueError("recourse needs at least one decision rule")
        unique(tuple(r.control for r in self.rules), "control permissions")
        if (
            self.mode == "fixed"
            and self.rules
            and not any(r.fixed_value is not None for r in self.rules)
        ):
            raise ValueError("fixed recourse must specify a fixed decision")

    def rule(self, control):
        for rule in self.rules:
            if rule.control == control:
                return rule
        raise ValueError(f"no permission for control: {control}")

    @property
    def ref(self):
        return reference(self, "recourse_policy/v1")
