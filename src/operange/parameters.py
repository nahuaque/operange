"""Parameter values, covariance and joint samples independent of their provider.

Coordinates and native units belong to the data. Component paths belong to
explicit model bindings. Sample moments do not imply a probability law or
confidence region; bounded uncertainty remains a separate domain declaration.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from .contract_types import Record, freeze, nonempty, unique
from .domains import ParameterSpace, FiniteSet, Scenario


def covariance_matrix(covariance, size):
    """Validate in correlation coordinates so units cannot hide indefiniteness."""
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError("Covariance must be a finite square matrix in parameter order")
    diagonal = np.diag(matrix)
    if np.any(diagonal < 0):
        raise ValueError("Covariance variances must be nonnegative")
    zero = diagonal == 0
    if np.any(matrix[zero, :] != 0) or np.any(matrix[:, zero] != 0):
        raise ValueError("Zero variance requires zero cross-covariances")
    positive = ~zero
    if np.any(positive):
        scales = np.sqrt(diagonal[positive])
        correlation = matrix[np.ix_(positive, positive)] / scales[:, None] / scales
        if not np.allclose(correlation, correlation.T, rtol=0, atol=1e-12):
            raise ValueError("Covariance must be symmetric")
        if np.linalg.eigvalsh((correlation + correlation.T) / 2).min() < -1e-12:
            raise ValueError("Covariance must be positive semidefinite")
    return (matrix + matrix.T) / 2


@dataclass(frozen=True)
class ParameterPoint(Record):
    space: ParameterSpace
    values: Mapping[str, float]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def _validate(self):
        self.space.validate(self.values)


@dataclass(frozen=True)
class ParameterBinding(Record):
    """Map one named coordinate to a component in a particular model."""

    parameter: str
    component: str
    unit: str

    def _validate(self):
        for name in ("parameter", "component", "unit"):
            nonempty(getattr(self, name), name)


@dataclass(frozen=True)
class ParameterCovariance(Record):
    space: ParameterSpace
    covariance: tuple[tuple[float, ...], ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def _validate(self):
        covariance_matrix(self.covariance, len(self.space.names))

    def in_space(self, space):
        """Align by coordinate identity and units, never by incidental order."""
        self.space.check_alignment(space)
        order = [self.space.names.index(name) for name in space.names]
        return tuple(tuple(self.covariance[i][j] for j in order) for i in order)


@dataclass(frozen=True)
class ParameterSamples(Record):
    """Labelled whole parameter vectors; duplicate values retain multiplicity."""

    space: ParameterSpace
    samples: tuple[Scenario, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def _validate(self):
        if not self.samples:
            raise ValueError("Parameter samples must contain at least one vector")
        unique(tuple(s.name for s in self.samples), "sample names")
        for sample in self.samples:
            self.space.validate(sample.values)

    def as_domain(self):
        return FiniteSet(self.space, self.samples)

    @cached_property
    def statistics(self):
        if len(self.samples) < 2:
            raise ValueError("Sample covariance requires at least two vectors")
        rows = np.asarray(
            [[s.values[n] for n in self.space.names] for s in self.samples]
        )
        mean = rows.mean(axis=0)
        centered = rows - mean
        covariance = centered.T @ centered / (len(rows) - 1)
        return freeze(
            {
                "parameters": self.space.names,
                "units": tuple(c.unit for c in self.space.coordinates),
                "mean": mean.tolist(),
                "covariance": covariance.tolist(),
                "samples": len(rows),
                "interpretation": "Empirical sample moments only; no Gaussian model or confidence region inferred.",
            }
        )

    @cached_property
    def covariance(self):
        return ParameterCovariance(
            self.space,
            self.statistics["covariance"],
            {
                **dict(self.provenance),
                "method": "empirical_sample_covariance",
                "samples": len(self.samples),
            },
        )
