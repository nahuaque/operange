"""Reuse IDAES initialization and diagnostics without changing study equations."""

from io import StringIO

from ..contract_types import Evidence, plain
from ._idaes_model import backend, component, number, uid


class InitializationFailure(RuntimeError):
    def __init__(self, message, code="idaes_initialization_failed"):
        super().__init__(message)
        self.code = code


def initialize(model, adapter, evidence):
    """Run declared square-unit steps, retaining only numerical starting values.

    Service constraints are excluded on the disposable initialization clone.
    Adjustable commands are temporarily fixed at their declared source guesses.
    No parameters, bounds, fixed values, equations or activation flags are copied
    back, even when an upstream initializer fails to restore its own model.
    """
    if not adapter.initialization:
        return
    pyo, _ = backend()
    from idaes.core.util.model_statistics import degrees_of_freedom
    from idaes.core.initialization import InitializationStatus

    working = model.clone()
    for req in adapter.requirements:
        component(working, req.component).deactivate()
    try:
        for binding in adapter.controls:
            var = component(working, binding.component)
            if not var.fixed:
                var.fix(number(var))
    except Exception as exc:
        evidence.append(
            Evidence(
                "initialization_setup",
                "initialization",
                "idaes_initialization",
                "unresolved",
                details={"error": str(exc)},
            )
        )
        raise InitializationFailure(
            "Initialization needs a finite source guess for every adjustable actuator.",
            "idaes_initialization_specification",
        ) from exc
    for index, step in enumerate(adapter.initialization):
        block = component(working, step.component)
        details = {
            "component": step.component,
            "method": step.method,
            "options": step.options,
            "scope": "Starting values only; service requirements temporarily excluded and adjustable actuators held at source guesses.",
        }
        try:
            dof = degrees_of_freedom(block)
            details["degrees_of_freedom"] = dof
            if dof != 0:
                raise InitializationFailure(
                    f"{step.component} has {dof} degrees of freedom during initialization; "
                    "supply a square unit specification, including its inlet conditions.",
                    "idaes_initialization_specification",
                )
            if step.method == "block":
                result = block.initialize(**plain(step.options))
                if result is not None and not (
                    result is InitializationStatus.Ok
                    or (
                        hasattr(result, "solver")
                        and pyo.check_optimal_termination(result)
                    )
                ):
                    raise InitializationFailure(
                        f"Upstream initialization did not report success: {result}"
                    )
                details["status"] = "completed" if result is None else str(result)
            else:
                initializer = block.default_initializer(**plain(step.options))
                details["initializer"] = (
                    f"{type(initializer).__module__}.{type(initializer).__qualname__}"
                )
                result = initializer.initialize(block)
                details["status"] = str(result)
                if result is not InitializationStatus.Ok:
                    raise InitializationFailure(
                        f"Upstream initialization returned {result}"
                    )
        except Exception as exc:
            details["error"] = f"{type(exc).__name__}: {exc}"
            evidence.append(
                Evidence(
                    f"initialization:{index}",
                    step.component,
                    "idaes_initialization",
                    "unresolved",
                    details=details,
                )
            )
            if isinstance(exc, InitializationFailure):
                raise
            raise InitializationFailure(
                f"Initialization of {step.component} failed: {type(exc).__name__}: {exc}"
            ) from exc
        evidence.append(
            Evidence(
                f"initialization:{index}",
                step.component,
                "idaes_initialization",
                "verified",
                details=details,
            )
        )
    # The model and its independent checker retain their original semantics.
    for var in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        if not var.fixed:
            source = component(working, uid(var))
            if source.value is not None:
                var.set_value(number(source), skip_validation=True)


def diagnostics(model, adapter, evidence, *, stage, rows, iterate_origin):
    """Advisory upstream diagnostics; unavailable diagnostics never decide a claim."""
    if not adapter.diagnostics:
        return
    pyo, _ = backend()
    from idaes.core.util.model_statistics import degrees_of_freedom
    from idaes.core.util.scaling import (
        badly_scaled_var_generator,
        unscaled_variables_generator,
    )

    details = {
        "stage": stage,
        "iterate_origin": iterate_origin,
        "scope": "Advisory checks at the current iterate, not a feasibility or infeasibility certificate. Nonzero optimization degrees of freedom can be intentional.",
    }
    try:
        # The public toolbox moved in IDAES 2.12; keep the declared 2.10+ range.
        try:
            from idaes.core.util.diagnostics_tools import DiagnosticsToolbox
        except ImportError:
            from idaes.core.util.model_diagnostics import DiagnosticsToolbox
        details["degrees_of_freedom"] = degrees_of_freedom(model)
        details["uninitialized_variables"] = [
            uid(v)
            for v in model.component_data_objects(pyo.Var, active=True)
            if v.value is None
        ]
        details["unscaled_variables"] = [
            uid(v) for v in unscaled_variables_generator(model)
        ]
        details["badly_scaled_variables"] = [
            {"component": uid(v), "scaled_magnitude": number(value)}
            for v, value in badly_scaled_var_generator(model)
        ]
        stream = StringIO()
        toolbox = DiagnosticsToolbox(model)
        toolbox.report_structural_issues(stream=stream)
        details["structural_report"] = stream.getvalue()
        details["structural_partition"] = {
            name: [[uid(item) for item in block] for block in blocks]
            for name, blocks in zip(
                (
                    "underconstrained_variables",
                    "underconstrained_constraints",
                    "overconstrained_variables",
                    "overconstrained_constraints",
                ),
                toolbox.get_dulmage_mendelsohn_partition(),
            )
        }
        if stage != "before_initialization":
            # The Jacobian checks need PyNumero's optional ASL library. Avoid
            # constructing its interface when absent (including destructor noise).
            from pyomo.contrib.pynumero.asl import AmplInterface

            if AmplInterface.available():
                stream = StringIO()
                try:
                    toolbox.report_numerical_issues(stream=stream)
                    details["numerical_report"] = stream.getvalue()
                except Exception as exc:
                    details["numerical_report_unavailable"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            else:
                details["numerical_report_unavailable"] = (
                    "PyNumero ASL library is unavailable; variable scaling and source residual checks still run."
                )
        violations, undefined = [], []
        for row in rows:
            try:
                value = number(row.expression(model))
                if (abs(value) if row.side == "eq" else value) > row.tolerance:
                    violations.append(
                        {
                            "constraint": row.name,
                            "component": row.path,
                            "kind": row.kind,
                            "residual": value,
                            "unit": row.unit,
                            "tolerance": row.tolerance,
                        }
                    )
            except Exception as exc:
                undefined.append({"constraint": row.name, "error": str(exc)})
        details["violated_rows_at_iterate"] = violations
        details["undefined_rows_at_iterate"] = undefined
        outcome = "verified"
    except Exception as exc:
        details["error"] = f"{type(exc).__name__}: {exc}"
        outcome = "unresolved"
    evidence.append(
        Evidence(
            f"diagnostics:{stage}",
            "model_diagnostics",
            "idaes_diagnostics_toolbox",
            outcome,
            backend="idaes",
            details=details,
        )
    )
