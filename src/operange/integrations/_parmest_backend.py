"""Thin parmest execution bridge and checks against original experiment models."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isclose
from types import SimpleNamespace

from ..contract_types import Diagnostic, digest, nonempty, plain
from ..primitives import finite
from ._idaes_model import (
    backend,
    component,
    model_manifest,
    number,
    prepare_template,
    uid,
    unit,
)


def parmest_backend():
    pyo, versions = backend()
    try:
        import pandas
        from pyomo.contrib.parmest.parmest import Estimator
        from pyomo.contrib.parmest.experiment import Experiment
    except ImportError as exc:
        raise ImportError("Install operange[idaes] to run parmest calibration") from exc
    return pyo, Estimator, Experiment, {**versions, "pandas": pandas.__version__}


@dataclass(frozen=True)
class ExperimentSnapshot:
    name: str
    model: object
    rows: tuple
    observations: tuple
    parameters: tuple
    variable_groups: tuple


def _scalar_label(model, obj, kinds):
    if not hasattr(obj, "ctype") or obj.is_indexed() or obj.ctype not in kinds:
        raise ValueError(
            "Calibration labels must reference supported scalar components"
        )
    if component(model, uid(obj)) is not obj:
        raise ValueError(
            "Calibration labels reference a component outside the experiment"
        )


def prepare_study(study, experiments):
    pyo, _, _, versions = parmest_backend()
    from .idaes import IdaesVariable

    if not isinstance(experiments, Mapping) or not experiments:
        raise ValueError("experiments must be a nonempty name-to-Experiment mapping")
    snapshots, manifests, signature = [], [], None
    for name, experiment in experiments.items():
        if not isinstance(name, str):
            raise ValueError("experiment names must be strings")
        nonempty(name, "experiment name")
        if not callable(getattr(experiment, "get_labeled_model", None)):
            raise ValueError("Supply upstream Experiments with get_labeled_model()")
        source = experiment.get_labeled_model()
        if (
            not isinstance(source, pyo.ConcreteModel)
            or source.parent_block() is not None
        ):
            raise ValueError("Experiments must return root ConcreteModels")
        model = source.clone()
        for suffix in ("experiment_outputs", "unknown_parameters"):
            if not isinstance(getattr(model, suffix, None), pyo.Suffix):
                raise ValueError(f"Experiment {name} needs the {suffix} suffix")
        if not len(model.experiment_outputs):
            raise ValueError("Every experiment needs at least one measured output")
        if hasattr(model, "measurement_error") and not isinstance(
            model.measurement_error, pyo.Suffix
        ):
            raise ValueError("measurement_error must be a Pyomo Suffix")
        observations = []
        for obj, measured in model.experiment_outputs.items():
            _scalar_label(model, obj, (pyo.Var, pyo.Param, pyo.Expression))
            sigma = None
            if hasattr(model, "measurement_error") and obj in model.measurement_error:
                sigma = finite(
                    model.measurement_error[obj], "measurement standard deviation"
                )
                if sigma <= 0:
                    raise ValueError("measurement standard deviation must be positive")
            if study.objective == "SSE_weighted" and sigma is None:
                raise ValueError(
                    "Weighted SSE requires measurement_error for every observation"
                )
            observations.append(
                {
                    "component": uid(obj),
                    "unit": unit(obj),
                    "measured": finite(measured, "measurement"),
                    "standard_deviation": sigma,
                }
            )
        if study.objective == "SSE" and len({o["unit"] for o in observations}) != 1:
            raise ValueError(
                "Unweighted SSE requires a common native observation unit; use weighted SSE"
            )
        parameters = []
        for obj in model.unknown_parameters:
            _scalar_label(model, obj, (pyo.Var, pyo.Param))
            if obj.ctype is pyo.Param and obj.parent_component().is_indexed():
                raise ValueError(
                    "parmest cannot estimate indexed Param elements; use scalar Params or scalar Var labels"
                )
            if obj.ctype is pyo.Param and (
                not obj.parent_component().mutable
                or obj.parent_component().domain not in (pyo.Reals, pyo.Any)
                or obj.parent_component()._validate is not None
            ):
                raise ValueError(
                    "Estimated Params must be mutable Reals/Any without validation callbacks"
                )
            parameters.append(
                {
                    "component": uid(obj),
                    "unit": unit(obj),
                    "initial_value": number(obj),
                    "lower": None
                    if obj.ctype is pyo.Param or obj.lower is None
                    else number(obj.lower),
                    "upper": None
                    if obj.ctype is pyo.Param or obj.upper is None
                    else number(obj.upper),
                }
            )
        current = {p["component"]: p["unit"] for p in parameters}
        if signature is not None and current != signature:
            raise ValueError(
                "Every experiment must label the same shared parameters with matching native units"
            )
        signature = current
        proxy = SimpleNamespace(
            inputs=(),
            controls=(),
            requirements=(),
            initialization=(),
            outputs=tuple(
                IdaesVariable(f"observation:{i}", o["component"], o["unit"])
                for i, o in enumerate(observations)
            ),
            tolerance=study.tolerance,
        )
        template, rows, _ = prepare_template(proxy, model)
        active_parameters = {
            uid(obj)
            for kind in (pyo.Var, pyo.Param)
            for obj in template.component_data_objects(kind, active=True)
            if hasattr(obj, "ctype")
        }
        if not set(current).issubset(active_parameters):
            raise ValueError(
                "Estimated parameters must belong to active experiment blocks"
            )
        variables = tuple(
            (uid(group), tuple(uid(v) for v in group.values()))
            for group in template.component_objects(pyo.Var, active=True)
        )
        source_manifest = model_manifest(template)
        manifests.append(
            {
                "name": name,
                "source": source_manifest,
                "source_fingerprint": digest(source_manifest),
                "observations": observations,
                "parameters": parameters,
                "checks": [
                    {
                        "component": r.path,
                        "side": r.side,
                        "unit": r.unit,
                        "tolerance": r.tolerance,
                    }
                    for r in rows
                ],
            }
        )
        snapshots.append(
            ExperimentSnapshot(
                name, template, rows, tuple(observations), tuple(parameters), variables
            )
        )
    if study.objective == "SSE":
        if len({o["unit"] for s in snapshots for o in s.observations}) != 1:
            raise ValueError(
                "Unweighted SSE requires common observation units across experiments"
            )
    manifest = {
        "name": study.name,
        "objective": study.objective,
        "objective_aggregation": "parmest mean of experiment objectives",
        "solver": "ipopt",
        "solver_interface": "ef_ipopt",
        "solver_options": study.solver_options,
        "constraint_tolerance": study.tolerance,
        "variable_bound_tolerance": 0,
        "versions": versions,
        "experiments": manifests,
        "provenance": study.provenance,
        "scope": "Numerical fit and source residual checks; no global optimality, independent certificate, or physical accuracy guarantee.",
        "identifiability": "not_assessed",
        "parameter_uncertainty": "not_computed",
        "fixed_conditions": "All fixed source variables remain fixed except explicitly labelled shared parameters.",
        "constraints": "All active experiment constraints are retained; source objectives are replaced by the chosen least-squares objective.",
    }
    return tuple(snapshots), manifest


def _estimate(study):
    """Use only the public Experiment/Estimator API; never inspect its EF model."""
    pyo, Estimator, Experiment, _ = parmest_backend()
    if not pyo.SolverFactory("ipopt").available(exception_flag=False):
        raise FileNotFoundError(
            "Ipopt is unavailable; install its executable separately"
        )

    # Upstream may request experiments repeatedly. Each request returns a clone
    # of the already captured source, never a fresh call to user model builders.
    class FrozenExperiment(Experiment):
        def get_labeled_model(self):
            return self.model.clone()

    estimator = Estimator(
        [FrozenExperiment(s.model.clone()) for s in study._experiments],
        obj_function=study.objective,
        solver_options=plain(study.solver_options),
    )
    # theta_est's public return_values accepts component-level indexed Vars;
    # scalar VarData paths are not supported by its extraction code.
    names = list(
        dict.fromkeys(name for s in study._experiments for name, _ in s.variable_groups)
    )
    # An experiment can contain only expressions of estimated scalar Params.
    # parmest converts these to Vars. Include them so its returned table keeps
    # that experiment's row even when other experiments have additional states.
    names = list(
        dict.fromkeys(
            names
            + [
                p["component"]
                for s in study._experiments
                for p in s.parameters
                if component(s.model, p["component"]).ctype is pyo.Param
            ]
        )
    )
    result = estimator.theta_est(return_values=names)
    if names:
        objective, estimates, frame = result
        if len(frame) != len(study._experiments):
            raise ValueError("parmest returned an incomplete experiment table")
        returned = frame.to_dict(orient="records")
    else:
        objective, estimates = result
        returned = [{} for _ in study._experiments]
    return objective, estimates, returned


def _check_experiment(snapshot, estimates, returned):
    from .parmest import (
        CalibrationCheck,
        CalibrationExperimentResult,
        CalibrationObservation,
    )

    pyo, _ = backend()
    checker = snapshot.model.clone()
    parameter_names = {p["component"] for p in snapshot.parameters}
    for name in parameter_names:
        component(checker, name).set_value(estimates[name])
    values = {}
    for group, paths in snapshot.variable_groups:
        if group not in returned:
            raise ValueError(f"Missing returned variable group: {group}")
        data = returned[group]
        data = [data] if len(paths) == 1 else data
        if not isinstance(data, (list, tuple)) or len(data) != len(paths):
            raise ValueError(f"Wrong returned variable group shape: {group}")
        for path, raw in zip(paths, data):
            value = finite(raw, "returned variable")
            var = component(checker, path)
            if path in parameter_names:
                if not isclose(value, estimates[path], rel_tol=1e-10, abs_tol=1e-8):
                    raise ValueError(
                        f"Experiment disagrees with shared parameter: {path}"
                    )
                value = estimates[path]
            elif var.fixed and value != number(var):
                raise ValueError(
                    f"parmest changed a fixed experimental condition: {path}"
                )
            var.set_value(value, skip_validation=True)
            values[path] = value
    observations = tuple(
        CalibrationObservation(**o, fitted=number(component(checker, o["component"])))
        for o in snapshot.observations
    )
    checks = tuple(
        CalibrationCheck(
            row.path, row.side, row.unit, number(row.expression(checker)), row.tolerance
        )
        for row in snapshot.rows
    )
    return CalibrationExperimentResult(snapshot.name, observations, checks, values)


def fit(study):
    from .parmest import CalibrationEstimate, CalibrationResult
    from pyomo.core.base.componentuid import ComponentUID

    solver = {
        "name": "ipopt",
        "interface": "parmest.Estimator.theta_est",
        "termination": None,
        "termination_scope": "The public theta_est return tuple does not expose raw solver termination; normal return and source candidate checks are recorded separately.",
    }
    diagnostics, estimates, experiments, objective = [], (), (), None
    stage = "solve"
    try:
        objective_raw, theta, returned = _estimate(study)
        solver["api_outcome"] = "returned"
        solver["reported_objective"] = finite(objective_raw, "parmest objective")
        stage = "candidate_check"
        canonical = {}
        for name, value in theta.items():
            path = str(ComponentUID(str(name)))
            if path in canonical:
                raise ValueError("Duplicate parameter aliases returned by parmest")
            canonical[path] = finite(value, "parameter estimate")
        expected = {p["component"] for p in study._experiments[0].parameters}
        if set(canonical) != expected:
            raise ValueError(
                "parmest returned missing or unexpected parameter estimates"
            )
        if len(returned) != len(study._experiments):
            raise ValueError("parmest returned an incomplete experiment table")
        experiments = tuple(
            _check_experiment(s, canonical, r)
            for s, r in zip(study._experiments, returned)
        )
        if any(not c.satisfied for e in experiments for c in e.checks):
            raise ValueError(
                "Returned calibration candidate violates a source constraint or bound"
            )
        estimates_list = []
        for p in study._experiments[0].parameters:
            peers = [
                q
                for s in study._experiments
                for q in s.parameters
                if q["component"] == p["component"]
            ]
            lower = max(
                (q["lower"] for q in peers if q["lower"] is not None), default=None
            )
            upper = min(
                (q["upper"] for q in peers if q["upper"] is not None), default=None
            )
            estimates_list.append(
                CalibrationEstimate(
                    p["component"], p["unit"], canonical[p["component"]], lower, upper
                )
            )
        # Re-evaluate the same upstream objective on original observation values.
        # Weighted SSE in current parmest includes a factor of 1/2; using its
        # public objective builder avoids mirroring a version-specific formula.
        from pyomo.contrib.parmest.parmest import SSE, SSE_weighted

        objective_fn = SSE_weighted if study.objective == "SSE_weighted" else SSE
        totals = []
        for s, result in zip(study._experiments, experiments):
            m = s.model.clone()
            for path, value in canonical.items():
                component(m, path).set_value(value)
            for path, value in result.values.items():
                component(m, path).set_value(value, skip_validation=True)
            totals.append(number(objective_fn(m)))
        checked_objective = sum(totals) / len(totals)
        if not isclose(
            solver["reported_objective"], checked_objective, rel_tol=1e-7, abs_tol=1e-8
        ):
            raise ValueError(
                "parmest objective disagrees with returned measurement residuals"
            )
        objective = checked_objective
        estimates = tuple(estimates_list)
        solver["candidate_checks"] = "passed"
        for p in estimates:
            if any(
                bound is not None
                and isclose(p.value, bound, rel_tol=1e-7, abs_tol=1e-8)
                for bound in (p.lower, p.upper)
            ):
                diagnostics.append(
                    Diagnostic(
                        "parameter_at_bound",
                        p.component,
                        "Estimate is at or near a declared bound; uncertainty and identifiability have not been assessed.",
                    )
                )
        return CalibrationResult(
            study.manifest,
            "completed",
            estimates,
            experiments,
            objective,
            solver,
            tuple(diagnostics),
        )
    except Exception as exc:
        solver.setdefault("api_outcome", "raised")
        code = (
            "calibration_solver_unavailable"
            if isinstance(exc, FileNotFoundError)
            else f"calibration_{stage}_failed"
        )
        diagnostics.append(Diagnostic(code, stage, f"{type(exc).__name__}: {exc}"))
        return CalibrationResult(
            study.manifest,
            "unresolved",
            (),
            experiments,
            None,
            solver,
            tuple(diagnostics),
        )


def apply_calibration(result, adapter, parameter_map):
    from .idaes import IdaesAdapter

    if result.execution != "completed":
        raise ValueError("Only completed, checked calibration results can be applied")
    if not isinstance(adapter, IdaesAdapter):
        raise ValueError("Apply calibration to an IdaesAdapter")
    if not isinstance(parameter_map, Mapping) or not parameter_map:
        raise ValueError(
            "Supply an explicit estimate-CUID to target-CUID parameter_map"
        )
    estimates = {p.component: p for p in result.estimates}
    if set(parameter_map) != set(estimates):
        raise ValueError("parameter_map must cover exactly the fitted parameters")
    pyo, _ = backend()
    model = adapter._template.clone()
    forbidden = {
        id(component(model, b.component)) for b in adapter.inputs + adapter.controls
    }
    active = {
        id(obj)
        for kind in (pyo.Var, pyo.Param)
        for obj in model.component_data_objects(kind, active=True)
    }
    updates, targets = [], set()
    for source, path in parameter_map.items():
        if not isinstance(path, str):
            raise ValueError("Target component paths must be strings")
        target = component(model, path)
        if id(target) not in active:
            raise ValueError("Calibration targets must belong to active model blocks")
        if id(target) in forbidden:
            raise ValueError(
                "Calibration targets cannot be scenario inputs or actuator controls"
            )
        if id(target) in targets:
            raise ValueError("Calibration parameter targets cannot alias each other")
        targets.add(id(target))
        estimate = estimates[source]
        if unit(target) != estimate.unit:
            raise ValueError("Calibration target native units must match the estimate")
        if target.ctype is pyo.Var:
            if not target.fixed:
                raise ValueError(
                    "Calibration target Vars must be fixed model parameters"
                )
            if (target.lb is not None and estimate.value < number(target.lb)) or (
                target.ub is not None and estimate.value > number(target.ub)
            ):
                raise ValueError("Calibrated parameter violates target bounds")
        elif target.ctype is not pyo.Param or not target.parent_component().mutable:
            raise ValueError("Calibration targets must be fixed Vars or mutable Params")
        elif (
            target.parent_component().domain not in (pyo.Reals, pyo.Any)
            or target.parent_component()._validate is not None
        ):
            raise ValueError(
                "Target Params must use Reals/Any without validation callbacks"
            )
        previous = number(target)
        target.set_value(estimate.value)
        updates.append(
            {
                "estimate": source,
                "target": uid(target),
                "unit": estimate.unit,
                "previous": previous,
                "value": estimate.value,
            }
        )
    provenance = {
        **plain(adapter.provenance),
        "calibration": {
            "result_ref": result.ref.to_dict(),
            "study_ref": reference_study(result.study),
            "parent_contract_ref": adapter.base_contract.ref.to_dict(),
            "updates": updates,
            "scope": "Fitted point estimates only; parameter uncertainty has not been propagated.",
        },
    }
    return replace(adapter, model=model, provenance=provenance)


def reference_study(study):
    from ..contract_types import reference

    return reference(study, "calibration_study/v1").to_dict()
