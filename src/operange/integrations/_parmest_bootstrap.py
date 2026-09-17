"""Upstream joint bootstrap sampling and checked fixed-parameter replays."""

from dataclasses import replace
from threading import Lock

from ..contract_types import Diagnostic
from ..primitives import finite
from ._idaes_model import component
from ._parmest_backend import estimator, parmest_backend

_SAMPLING_LOCK = Lock()


def sample(study, count, seed):
    import numpy as np

    # parmest's public bootstrap currently seeds NumPy's legacy global RNG.
    # Keep this call from altering the caller's subsequent random sequence.
    with _SAMPLING_LOCK:
        state = np.random.get_state()
        try:
            return (
                estimator(study)
                .theta_est_bootstrap(
                    count,
                    samplesize=len(study._experiments),
                    replacement=True,
                    seed=seed,
                    return_samples=True,
                )
                .to_dict(orient="records")
            )
        finally:
            np.random.set_state(state)


def replay(study, fitted, draw, indices, estimates):
    from .parmest import ParmestCalibration

    pyo, _, Experiment, _ = parmest_backend()
    experiments = {}
    for position, index in enumerate(indices):
        source = study._experiments[index]
        model = source.model.clone()
        for estimate in estimates:
            target = component(model, estimate.component)
            if target.ctype is pyo.Var:
                target.fix(estimate.value)
            else:
                target.set_value(estimate.value)
        model.unknown_parameters.clear()
        experiments[f"{position}:{source.name}"] = Experiment(model)
    return ParmestCalibration(
        name=f"{study.name}: fixed bootstrap draw {draw}",
        experiments=experiments,
        objective=study.objective,
        solver_options=study.solver_options,
        tolerance=study.tolerance,
        provenance={
            "bootstrap_calibration": fitted.ref.to_dict(),
            "bootstrap_draw": draw,
            "resampled_experiments": [study._experiments[i].name for i in indices],
            "scope": "Physical replay at the supplied joint vector; bootstrap optimality and identifiability are not assessed.",
        },
    ).fit()


def bootstrap(study, *, samples, seed):
    from .parmest import CalibrationBootstrap, CalibrationDraw
    from pyomo.core.base.componentuid import ComponentUID

    if type(samples) is not int or samples < 2:
        raise ValueError("samples must be an integer of at least two")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    parameters = study._experiments[0].parameters
    if not parameters:
        raise ValueError(
            "Parameter uncertainty needs estimated parameters, not reconciliation alone"
        )
    if len(study._experiments) <= len(parameters):
        raise ValueError(
            "parmest bootstrap needs more experiments than estimated parameters"
        )
    if any(p["component"] == "samples" for p in parameters):
        raise ValueError(
            "parmest reserves the parameter name 'samples' for resampling identities"
        )
    fitted = study.fit()
    if fitted.execution != "completed":
        return CalibrationBootstrap(
            fitted,
            samples,
            seed,
            "unresolved",
            diagnostics=(
                Diagnostic(
                    "bootstrap_calibration_unresolved",
                    "calibration",
                    "Joint uncertainty requires a completed calibration first.",
                ),
            ),
        )
    diagnostics, draws = [], []
    try:
        rows = sample(study, samples, seed)
        if len(rows) != samples:
            raise ValueError("parmest returned an incomplete bootstrap table")
    except Exception as exc:
        return CalibrationBootstrap(
            fitted,
            samples,
            seed,
            "unresolved",
            diagnostics=(
                Diagnostic(
                    "bootstrap_sampling_failed",
                    "parmest",
                    f"{type(exc).__name__}: {exc}. Upstream exposes no partial table after a failed batch.",
                ),
            ),
        )

    expected = {p.component for p in fitted.estimates} | {"samples"}
    for index, row in enumerate(rows):
        name = f"draw-{index:06d}"
        names, estimates, checked, errors = (), (), None, []
        try:
            canonical = {}
            for key, value in row.items():
                path = "samples" if key == "samples" else str(ComponentUID(str(key)))
                if path in canonical:
                    raise ValueError("Duplicate bootstrap parameter aliases")
                canonical[path] = value
            row = canonical
            if set(row) != expected:
                raise ValueError(
                    "Bootstrap row has missing or unexpected parameter coordinates"
                )
            indices = tuple(row["samples"])
            if len(indices) != len(study._experiments) or any(
                isinstance(i, bool)
                or int(i) != i
                or not 0 <= int(i) < len(study._experiments)
                for i in indices
            ):
                raise ValueError("Invalid bootstrap resampling identities")
            indices = tuple(int(i) for i in indices)
            if len(set(indices)) <= len(parameters):
                raise ValueError(
                    "Bootstrap row violates upstream distinct-experiment sampling rule"
                )
            names = tuple(study._experiments[i].name for i in indices)
            estimates = tuple(
                replace(p, value=finite(row[p.component], "bootstrap estimate"))
                for p in fitted.estimates
            )
            checked = replay(study, fitted, name, indices, estimates)
            if checked.execution != "completed":
                errors.append(
                    Diagnostic(
                        "bootstrap_replay_unresolved",
                        name,
                        "Fixed-parameter physical replay failed; this draw is retained and cannot be propagated.",
                    )
                )
        except Exception as exc:
            errors.append(
                Diagnostic(
                    "bootstrap_draw_invalid", name, f"{type(exc).__name__}: {exc}"
                )
            )
        draws.append(CalibrationDraw(name, names, estimates, checked, tuple(errors)))
    completed = all(d.checked for d in draws)
    if not completed:
        diagnostics.append(
            Diagnostic(
                "bootstrap_incomplete",
                "draws",
                "At least one draw is unresolved; no surviving subset is promoted to a complete joint uncertainty study.",
            )
        )
    return CalibrationBootstrap(
        fitted,
        samples,
        seed,
        "completed" if completed else "unresolved",
        tuple(draws),
        tuple(diagnostics),
    )
