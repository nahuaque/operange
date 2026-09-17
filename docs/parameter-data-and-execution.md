# Parameter data and checked model execution

Parameter data has a provider-independent representation in
`operange.parameters`, also exported from `operange`. It needs no IDAES, Pyomo,
parmest, pandas or solver installation.

| Type | Meaning |
| --- | --- |
| `ParameterPoint` | A coordinate space, one complete value vector, and provenance. |
| `ParameterBinding` | An explicit parameter name, target model component path, and native unit. |
| `ParameterCovariance` | A full covariance matrix in its declared coordinate order, with provenance. |
| `ParameterSamples` | Labelled whole vectors, preserving duplicates and provenance. |

These are immutable data records. Covariance is not a probability distribution
or confidence region. Samples are not converted into independent coordinate
intervals. Bounded uncertainty remains an explicit domain declaration.

## Supply parameter data from any source

```python
from operange import Coordinate, ParameterSpace, ParameterPoint, ParameterCovariance
from operange.integrations.idaes import IdaesVariable
from examples.local_uncertainty import operating_model

space = ParameterSpace((
    Coordinate("efficiency", "1", "External equipment study"),
    Coordinate("loss", "MW", "External equipment study"),
))
point = ParameterPoint(
    space, {"efficiency": 0.8, "loss": 1},
    provenance={"source": "External equipment study"},
)
covariance = ParameterCovariance(
    space, ((0.01, 0.04), (0.04, 0.25)),
    provenance={"source": "Full supplied covariance"},
)
adapter = operating_model().with_parameters(
    point,
    parameter_map={
        "efficiency": "fs.boiler.efficiency",
        "loss": "fs.boiler.loss",
    },
)
result = adapter.propagate_uncertainty(
    parameters=(
        IdaesVariable("efficiency", "fs.boiler.efficiency", "1"),
        IdaesVariable("loss", "fs.boiler.loss", "MW"),
    ),
    covariance=covariance,
    fixed_controls={"fuel": 15},
    margins=("service",),
)
```

`with_parameters` creates a fresh adapter snapshot and records the source data,
mapping, parent contract and assignments in provenance. It requires a complete
mapping. Target Vars must already be fixed parameters; Params must be mutable.
Targets must belong to active blocks, cannot alias each other, and cannot be
scenario inputs or actuator controls. Native units must match exactly.

Assignment applies the **whole vector before checking bounds**, including bounds
that depend on another assigned parameter. The original model is unchanged if
assignment fails. Binding and source ownership checks belong to the Pyomo
integration, rather than to the estimation provider.

A named `ParameterCovariance` is aligned to the requested parameter names and
units, even if its original order differs. Its original data and provenance are
retained under `result.provenance['parameter_covariance']`. Raw matrices remain
supported in the explicitly supplied parameter order.

## Reuse calibration results and samples

```python
point = fitted.parameter_point
samples = bootstrap.parameter_samples
moments = samples.statistics
covariance = samples.covariance
finite_domain = samples.as_domain()
```

The conversions require checked, completed calibration results. A reconciled
study with no estimated parameters has no `parameter_point`. Bootstrap conversion
requires every requested draw to be checked; it does not drop failed draws.

External producers can construct `ParameterSamples(space, samples, provenance)`
directly, with a tuple of existing `Scenario` records. Labels must be unique;
identical vectors with different labels remain distinct samples. Sample moments
require at least two vectors and use the full sample covariance with denominator
`n - 1`. The space's nominal point need not equal the empirical sample mean.

The existing calibration methods—`apply_to`, `as_domain`, `statistics`,
`as_claim`, `optimize_shared_controls`, and `propagate_uncertainty`—remain
available. They translate calibration data and provenance at the boundary,
then delegate assignment, sample composition or execution to shared helpers.
Existing calibration and operating-result schemas remain unchanged.

## Internal execution boundary

The internal implementation separates the following responsibilities:

1. `_idaes_parameters` binds neutral data onto a cloned source model.
2. `_idaes_execution` prepares a requested operating point, generates a solver
   candidate, and checks it against a separate source clone.
3. `_idaes_results` binds domain membership and formats the established portable
   evaluation records from the execution outcome.
4. Study code composes these operations into audits, shared dispatch, local
   sensitivities and joint-sample studies.

`OperatingPointRequest` snapshots input values, actuator commands/permissions,
and selected requirements. `PointExecution.prepare()` applies them and runs
the configured initialization. `generate_candidate()` uses the configured
solver. `check_candidate()` copies only returned variable values into the
unmodified source clone, verifies fixed values, and evaluates all included
constraints and bounds. A `CheckedPoint` carries that source model, values,
residuals and, when requested, the original objective value.

Local derivatives consume the checked runtime model directly. They do not
reconstruct a model from exported result values. Shared dispatch supplies typed
candidates to the same source checker, substituting the exact common commands
before acceptance. Solver failures, invalid candidates and source violations
retain their existing unresolved outcomes and diagnostics.

Calibration reuses parameter assignment and solver execution but retains its
measurement-specific checks. Held-out prediction still excludes measurement-fit
objectives and requires a regular equality-state model. Numerical source checks
do not establish nonlinear infeasibility or global optimality, and the independent
certificate verifier is unchanged.

The execution classes are private implementation interfaces; the public study
methods and portable data records are the supported application interface.
