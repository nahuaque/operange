"""Finite numerical evaluation; solver status never establishes infeasibility."""

from .._finite_audit import audit_finite
from ..claim import rejected_result
from ..contract_types import (
    ConstraintCheck,
    Diagnostic,
    EvaluationPayload,
    Evidence,
    Membership,
    QuantityValue,
    snapshot,
)
from ..engineering_results import EvaluationResult
from ._idaes_model import backend, component, number
from ._idaes_workflow import InitializationFailure, diagnostics, initialize


class SolverUnavailable(RuntimeError):
    pass


def _solve(model, adapter):
    pyo, _ = backend()
    from pyomo.common.modeling import unique_component_name

    # Some Pyomo solver wrappers return no solution object without an objective.
    # A constant zero objective preserves the supplied feasibility problem.
    if (
        next(
            model.component_data_objects(pyo.Objective, active=True, descend_into=True),
            None,
        )
        is None
    ):
        model.add_component(
            unique_component_name(model, "_operange_feasibility_objective"),
            pyo.Objective(expr=0),
        )
    solver = pyo.SolverFactory(adapter.solver)
    if not solver.available(exception_flag=False):
        raise SolverUnavailable(
            f"Solver {adapter.solver!r} is unavailable; install/configure it separately"
        )
    result = solver.solve(
        model, options=dict(adapter.solver_options), load_solutions=False
    )
    termination = str(result.solver.termination_condition)
    details = {
        "solver": adapter.solver,
        "status": str(result.solver.status),
        "termination": termination,
        "scope": "candidate generation only",
    }
    if termination not in {"optimal", "locallyOptimal", "globallyOptimal", "feasible"}:
        return False, details
    if not len(result.solution):
        return False, {**details, "reason": "No fresh solver solution was returned"}
    model.solutions.load_from(result)
    return True, details


def evaluate_result(claim, realization=None, *, contract=None):
    adapter = claim.adapter
    contract = claim.contract if contract is None else contract
    request = {"query": "evaluation", "realization": snapshot(realization)}
    try:
        point = adapter.input_space.validate(
            adapter.input_space.nominal if realization is None else realization
        )
        membership = claim.domain.membership(point)
    except (ValueError, TypeError) as exc:
        return rejected_result(
            contract,
            "evaluation",
            request,
            str(exc),
            code="invalid_realization",
            execution="invalid",
        )
    evidence = [
        Evidence(
            "membership",
            "domain_membership",
            "declared_domain_membership",
            "unresolved" if membership.status == "unknown" else "verified",
            details={"check": membership.to_dict()},
        )
    ]
    member = Membership(
        membership.status,
        evidence_refs=("membership",) if membership.status != "unknown" else (),
    )
    if membership.status != "inside":
        return EvaluationResult(
            contract,
            request,
            "invalid" if membership.status == "outside" else "unresolved",
            EvaluationPayload(
                "not_evaluated" if membership.status == "outside" else "unknown", member
            ),
            tuple(evidence),
            (
                Diagnostic(
                    "domain_membership",
                    "realization",
                    f"Membership is {membership.status}",
                ),
            ),
        )
    included = tuple(
        r
        for r in adapter._rows
        if r.kind != "requirement" or r.name in claim.requirements
    )
    values = tuple(QuantityValue(name, value) for name, value in point.items())
    checks = tuple(ConstraintCheck(row.name, "unknown") for row in included)
    feasible = False
    code = "idaes_evaluation_error"
    stage = "scenario_setup"
    model = None
    diagnostic_model = None
    iterate_origin = "source_start"
    message = "No checked numerical candidate was returned."
    try:
        pyo, _ = backend()
        model = adapter._template.clone()
        for binding in adapter.inputs:
            item = component(model, binding.component)
            if item.ctype is pyo.Var:
                item.fix(point[binding.name])
            else:
                item.set_value(point[binding.name])
        for binding in adapter.controls:
            item = component(model, binding.component)
            rule = claim.recourse.rule(binding.name)
            item.unfix() if rule.fixed_value is None else item.fix(rule.fixed_value)
        for req in adapter.requirements:
            if req.name not in claim.requirements:
                component(model, req.component).deactivate()
        diagnostics(
            model,
            adapter,
            evidence,
            stage="before_initialization",
            rows=included,
            iterate_origin=iterate_origin,
        )
        stage = "initialization"
        initialize(model, adapter, evidence)
        # Solver plugins may transform their model. Check the returned variables
        # against a separate, unmodified copy of the requested source equations.
        checker = model.clone()
        diagnostic_model = checker
        iterate_origin = "prepared_start_without_solver_solution"
        stage = "solve"
        solved, details = _solve(model, adapter)
        evidence.append(
            Evidence(
                "solver",
                "solver_termination",
                "pyomo_solver",
                "verified" if solved else "unresolved",
                backend=adapter.solver,
                details=details,
            )
        )
        if solved:
            stage = "candidate_check"
            from ._idaes_model import uid

            candidates = []
            for var in checker.component_data_objects(
                pyo.Var, active=True, descend_into=True
            ):
                candidate = number(component(model, uid(var)))
                if var.fixed and candidate != number(var):
                    raise ValueError(
                        f"Solver changed a fixed input, command or state: {uid(var)}"
                    )
                candidates.append((var, candidate))
            for var, candidate in candidates:
                var.set_value(candidate, skip_validation=True)
            model = checker
            iterate_origin = "returned_solver_candidate"
            candidate_values = [*values]
            candidate_values.extend(
                QuantityValue(v.name, number(component(model, v.component)))
                for v in adapter.controls + adapter.outputs
            )
            candidate_values.extend(
                QuantityValue(name, number(component(model, path)))
                for name, path, _ in adapter._states
            )
            residuals = [(row, number(row.expression(model))) for row in included]
            evidence.append(
                Evidence(
                    "residuals",
                    "numerical_feasibility",
                    "pyomo_expression_and_bound_checks",
                    "verified",
                    details={
                        "source_fingerprint": contract.model["source_fingerprint"],
                        "scope": "Floating-point substitution in all included source constraints and bounds; no global optimality or independent certificate.",
                        "residuals": {row.name: value for row, value in residuals},
                    },
                )
            )
            checks = tuple(
                ConstraintCheck(
                    row.name,
                    "satisfied"
                    if (abs(value) if row.side == "eq" else value) <= row.tolerance
                    else "violated",
                    QuantityValue(row.residual, value),
                    ("residuals",),
                )
                for row, value in residuals
            )
            values = tuple(candidate_values)
            feasible = all(check.assessment == "satisfied" for check in checks)
            code = "idaes_residual_violation"
            message = "Returned candidate violates a source constraint or bound; no infeasibility conclusion follows."
        else:
            code = "idaes_solver_failed"
            message = f"Solver termination {details['termination']}; no feasible candidate or infeasibility proof established."
    except Exception as exc:
        # External models and solver plugins can fail in many backend-specific ways.
        # Keep an unresolved scenario so a finite audit cannot silently pass.
        message = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, InitializationFailure):
            code = exc.code
        elif isinstance(exc, SolverUnavailable):
            code = "idaes_solver_unavailable"
        else:
            code = {
                "initialization": "idaes_initialization_failed",
                "solve": "idaes_solver_error",
                "candidate_check": "idaes_candidate_error",
            }.get(stage, "idaes_evaluation_error")
    if model is not None:
        diagnostics(
            diagnostic_model if diagnostic_model is not None else model,
            adapter,
            evidence,
            stage="checked_candidate" if feasible else f"unresolved_{stage}",
            rows=included,
            iterate_origin=iterate_origin,
        )
    return EvaluationResult(
        contract,
        request,
        "completed" if feasible else "unresolved",
        EvaluationPayload(
            "feasible" if feasible else "unknown",
            member,
            values,
            checks,
            included_constraints=tuple(row.name for row in included),
        ),
        tuple(evidence),
        () if feasible else (Diagnostic(code, stage, message),),
    )


def audit_result(claim):
    def no_failure(*args):
        raise AssertionError(
            "A numerical IDAES candidate cannot establish recourse infeasibility"
        )

    return audit_finite(
        claim,
        evaluate_result,
        no_failure,
        method="complete_finite_idaes_numerical_checks",
        scope={
            "coverage": "declared finite realizations only",
            "evidence": "numerical residual checks; not independently certified",
            "controls": "fixed as declared or independently adjustable after all inputs are observed",
        },
        diagnostic="A scenario lacks a checked numerical candidate; solver infeasibility is not a proof.",
    )
