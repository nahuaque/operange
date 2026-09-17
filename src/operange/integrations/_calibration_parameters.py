"""Convert checked calibration data to neutral parameter records."""

from ..domains import Coordinate, ParameterSpace, Scenario
from ..parameters import ParameterPoint, ParameterSamples


def estimate_point(estimates, *, provenance=None):
    return ParameterPoint(
        ParameterSpace(
            tuple(
                Coordinate(p.component, p.unit, "Parameter estimate", nominal=p.value)
                for p in estimates
            )
        ),
        {p.component: p.value for p in estimates},
        provenance or {},
    )


def bootstrap_samples(bootstrap):
    bootstrap._require_complete()
    result_id = bootstrap.result_id
    space = ParameterSpace(
        tuple(
            Coordinate(
                p.component, p.unit, "Joint bootstrap parameter", nominal=p.value
            )
            for p in bootstrap.calibration.estimates
        )
    )
    return ParameterSamples(
        space,
        tuple(
            Scenario(
                draw.name,
                {p.component: p.value for p in draw.estimates},
                f"Empirical joint draw from {result_id}; no confidence coverage asserted",
            )
            for draw in bootstrap.draws
        ),
        {
            "uncertainty_ref": bootstrap.ref.to_dict(),
            "calibration_ref": bootstrap.calibration.ref.to_dict(),
        },
    )


def calibration_updates(assignments):
    """Retain the established calibration provenance schema at its boundary."""
    return [
        {
            "estimate": a.parameter,
            "target": a.component,
            "unit": a.unit,
            "previous": a.previous,
            "value": a.value,
        }
        for a in assignments
    ]
