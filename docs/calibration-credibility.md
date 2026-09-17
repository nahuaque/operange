# Assess calibration credibility

A completed [parmest fit](calibration.md) establishes that its returned candidate
passed numerical source checks. The data may still leave parameter combinations
poorly determined, and the fitted model may predict new experiments badly.
These are separate questions with separate reports.

## Local parameter information

```python
fit = campaign.fit()
diagnosis = campaign.diagnose(
    fit,
    parameter_scales={"heat_transfer": 10, "area": 0.1},
)
if diagnosis.execution != "completed":
    raise RuntimeError(diagnosis.diagnostics)
print(dict(diagnosis.analysis))
```

The result is a `CalibrationDiagnostics`. Supply a positive scale for **every
estimated parameter**, in that parameter's declared native unit. A scale is a
chosen comparison increment, not a standard deviation or uncertainty bound.
Omitting `fit` runs `campaign.fit()` first; supplying it avoids another fit.
An existing result must belong to the exact captured campaign.

For each experiment, Operange rechecks the fitted variables against the source
model, fixes the estimated parameters and uses Pyomo's public
[automatic differentiation API](https://pyomo.readthedocs.io/en/stable/api/pyomo.core.expr.calculus.derivatives.differentiate.html).
For equality states `c(x, theta) = 0`, it computes
`dx/dtheta = -solve(c_x, c_theta)` and includes this state response in each
measured output's derivative. The normalized measurement matrix is

```text
J[i,j] = (dy[i]/dtheta[j]) * parameter_scale[j] / measurement_sigma[i]
```

Every measurement needs a positive `measurement_error`, including when the fit
used unweighted SSE. Correlated measurement errors are outside this slice.
The measurement-Jacobian view is also used by
[parmest's Gauss–Newton covariance method](https://pyomo.readthedocs.io/en/stable/explanation/analysis/parmest/covariance.html).
Here the matrix and its singular directions are retained without inverting a
potentially singular information matrix or inventing a confidence region.

`diagnosis.experiments` stores the dimensionless Jacobian rows in each source
experiment's declared observation order. Columns follow
`diagnosis.analysis["parameters"]`, matching `fit.estimates`. The analysis
reports singular values, numerical rank, a condition number for full-rank
matrices, and weak directions. Direction coefficients refer to
`z[j] = delta(theta[j]) / parameter_scale[j]`. Multiply each coefficient by its
parameter scale to express a physical perturbation. Signs are conventional;
bases within repeated singular subspaces are not unique.

`rank_tolerance` (default `1e-8`) and `weak_threshold` (default `1e-3`) are relative
to the largest singular value. Thus numerical rank and poor relative excitation
remain distinguishable. An all-zero matrix has rank zero and all directions
are numerically null. A singular condition number is represented by `None`.
Full rank does not establish precise estimates, structural identifiability,
global uniqueness, or accuracy away from the fitted point.

Parameters within `bound_tolerance` (default `1e-6`) of a bound, measured in
the supplied parameter scale, receive warnings. Experiment reports also list
active bounds and inequalities, using the larger of their declared residual
tolerance and `bound_tolerance` in the row's native unit. These limits are not
treated as measurements. Derivatives describe the ambient equality model;
they do not account for one-sided feasible directions or changing active sets.

After fixing parameters, the remaining states must form a square, numerically
regular equality system. The state Jacobian is equilibrated by row and column
magnitudes; condition numbers above `1e10` are unresolved. Free reconciliation
variables, parameter-only restrictions, nondifferentiable expressions, failed
source checks and missing measurement scales can also make an experiment
unresolved. This is a numerical check at the fitted point, not a proof of
regularity at an exact solution. Every experiment is retained, and aggregate
analysis is unavailable if any sensitivity is unresolved.

## Validate explicitly held-out experiments

```python
validation = campaign.validate(
    fit,
    experiments={"new-run-1": held_out_experiment_1,
                 "new-run-2": held_out_experiment_2},
    residual_threshold=3.0,
)
if validation.execution == "completed":
    print(validation.metrics)
for prediction in validation.predictions:
    print(prediction.name, prediction.execution, prediction.diagnostics)
```

Held-out experiments use the same upstream `Experiment` labels and parameter
paths/units as the training campaign. Names must be disjoint from training.
Each builder is captured once. The caller is responsible for genuinely holding
these data out; renamed training data cannot establish statistical independence.

The fitted parameters and recorded conditions stay fixed. All source objectives
are deactivated, and the prediction solve uses a constant feasibility objective.
Measurement labels never enter its objective. A structural matching check
rejects remaining free reconciliation variables before solving. Returned
variables are checked against separate source clones, and local state regularity
is checked at the candidate. Models with multiple nonlinear roots still need a
justified experimental initialization; global state uniqueness is not proved.

`CalibrationValidation.execution == "completed"` means all predictions were
computed and physically checked. **It is not an accuracy acceptance verdict.**
Each prediction retains measured and predicted values, native residuals,
standardized residuals where measurement errors are supplied, source constraints,
fixed conditions, solver details and warnings. Residuals are predicted minus
measured. A standardized residual exceeding `residual_threshold` produces a
diagnostic; the threshold is not a confidence test and does not incorporate
parameter uncertainty, measurement dependence or model discrepancy.

`metrics` groups observations by component path and native unit, reporting
count, bias, RMSE, maximum absolute residual and standardized RMSE. Unweighted
experiments without sigma still have physical residual metrics. A failed
prediction remains in the report and prevents aggregate metrics, rather than
silently dropping the difficult experiment. Missing solvers and infeasible
candidates remain unresolved numerical outcomes.

## Save and inspect without modeling dependencies

```python
from operange.integrations.parmest import CalibrationDiagnostics, CalibrationValidation

saved_diagnosis = diagnosis.to_json()
saved_validation = validation.to_json()
diagnosis = CalibrationDiagnostics.from_json(saved_diagnosis)
validation = CalibrationValidation.from_json(saved_validation)
```

`calibration_diagnostics/v1` stores the original fit, scales, thresholds,
per-experiment Jacobians, active-limit warnings and scope. Singular-value
summaries are computed from the stored matrix through `analysis`, including in
a base-only installation. `calibration_validation/v1` additionally retains the
held-out source declarations and every checked or unresolved prediction.
`metrics` can also be computed without Pyomo, IDAES, pandas or a solver.

Readers check content identity and record consistency. They do not execute
source equations or independently verify the nonlinear calculations. The
original `CalibrationResult` remains unchanged; its standalone scope is not
silently upgraded by these additional reports.

## Worked heat-transfer example

```bash
uv run --extra idaes python examples/calibration_credibility.py \
  --diagnostics-output local-information.json --validation-output validation.json
```

Synthetic data follow `duty = heat_transfer * area * temperature_difference`.
With known area, the coefficient has one measured local direction. Estimating
both coefficient and area gives rank one for two parameters and exposes the
combination that preserves their product. The known-area fit predicts two new
conditions without refitting; a deliberately biased held-out reading produces
an approximately 750 W residual and an explicit warning despite successful
physical prediction.
