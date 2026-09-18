"""Search for local parametric LP certificates without inferring solver guarantees."""

from fractions import Fraction as F

from ._cvxpy_backend import BackendUnavailable, validate_backend
from ._numeric import round_down
from ._parametric_dispatch import change_rates, problem
from .contract_types import Diagnostic, snapshot
from .dispatch_sensitivity import (
    DispatchChange,
    DispatchSensitivity,
    DispatchSensitivityResult,
    DispatchSlope,
)
from .linear_process import LinearProcessAdapter
from .objectives import LinearObjective


def analyze(claim, realization, changes, backend, max_refinements):
    request = snapshot(
        {
            "query": "dispatch_sensitivity",
            "realization": realization,
            "changes": [
                c.to_dict() if type(c) is DispatchChange else c for c in changes
            ]
            if isinstance(changes, (tuple, list))
            else changes,
            "backend": backend,
            "max_refinements": max_refinements,
            "scope": "ambient objective-value derivatives of real-valued optimized dispatch; one coordinate at a time",
        }
    )
    contract = claim.contract

    def rejected(message, execution="unsupported", base=None):
        return DispatchSensitivityResult(
            contract,
            request,
            execution,
            base=base,
            diagnostics=(
                Diagnostic("dispatch_sensitivity_" + execution, "sensitivity", message),
            ),
        )

    if type(claim.adapter) is not LinearProcessAdapter or not isinstance(
        claim.adapter.objective, LinearObjective
    ):
        return rejected(
            "Dispatch sensitivities require a declarative linear adapter with a linear objective and no saved controller."
        )
    model = claim.adapter
    try:
        model._permissions(claim)
    except ValueError as exc:
        return rejected(str(exc))
    if (
        len(model.variables) > 64
        or len(model.operating_limits)
        + len(model.requirements)
        + 4 * len(model.variables)
        > 4096
    ):
        return rejected(
            "Exact basis recovery is limited to 64 controls and 4096 physical rows."
        )
    try:
        if (
            not isinstance(changes, (tuple, list))
            or not changes
            or len(changes) > 64
            or any(type(c) is not DispatchChange for c in changes)
        ):
            raise ValueError(
                "provide between 1 and 64 declarative DispatchChange objects"
            )
        if len({c.change_id for c in changes}) != len(changes):
            raise ValueError("duplicate dispatch sensitivity changes")
        if type(max_refinements) is not int or not 0 <= max_refinements <= 32:
            raise ValueError("max_refinements must be an integer between 0 and 32")
        validate_backend(backend)
        point = model.input_space.validate(realization)
        system = problem(claim, point)
        derivatives = [change_rates(claim, system, change) for change in changes]
    except BackendUnavailable as exc:
        return rejected(str(exc))
    except (ValueError, TypeError, OverflowError) as exc:
        return rejected(str(exc), "invalid")
    base = claim.evaluate_result(point, backend=backend)
    if base.payload.feasibility != "feasible":
        return rejected(
            "Local slopes require a checked feasible base realization; inspect the retained evaluation.",
            "unresolved",
            base,
        )
    attempts = []

    def basis_at(rhs):
        try:
            basis, reports = system.proposal(rhs, model.solver_tolerance, backend)
            attempts.extend(reports)
            return basis
        except (ValueError, OverflowError, ZeroDivisionError, RuntimeError):
            return None

    basis = basis_at(system.rhs)
    sensitivities = []
    for change, (rates, constant) in zip(changes, derivatives):
        sides = []
        for side, direction in (("backward", -1), ("forward", 1)):
            unit = f"{model.output(model.objective.output).unit}/{change.unit}"
            proposal = basis
            found = None
            for iteration in range(max_refinements + 1):
                try:
                    segment = system.segment(
                        proposal, rates, constant, direction, change.step
                    )
                    if segment is not None:
                        radius, slope, proof = segment
                        exported_radius = round_down(radius)
                        if exported_radius > 0:
                            found = DispatchSlope(
                                side,
                                "verified",
                                unit,
                                str(slope),
                                exported_radius,
                                proof,
                            )
                            break
                    if iteration == max_refinements:
                        break
                    probe = direction * F(change.step) / 2**iteration
                    proposal = basis_at(
                        [b + probe * h for b, h in zip(system.rhs, rates)]
                    )
                    if proposal is None:
                        proof = system.infeasible_direction(
                            rates, direction, probe, model.solver_tolerance
                        )
                        if proof is not None:
                            found = DispatchSlope(
                                side,
                                "infeasible",
                                unit,
                                valid_step=change.step,
                                proof=proof,
                            )
                            break
                except (ValueError, OverflowError, ZeroDivisionError, RuntimeError):
                    proposal = None
            sides.append(found or DispatchSlope(side, "unresolved", unit))
        sensitivities.append(DispatchSensitivity(change, *sides))
    completed = all(s.resolution != "unresolved" for s in sensitivities)
    return DispatchSensitivityResult(
        contract,
        request,
        "completed" if completed else "unresolved",
        tuple(sensitivities),
        base,
        ()
        if completed
        else (
            Diagnostic(
                "local_basis_unresolved",
                "sensitivity",
                "Some directions have no checked local segment or exclusion certificate within the search budget; numerical status alone supplies no slope.",
            ),
        ),
        tuple(attempts),
    )
