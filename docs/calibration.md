# Calibrate existing models with parmest

`operange.integrations.parmest` is the public calibration integration for
IDAES/Pyomo models. It uses upstream `Experiment` and `Estimator.theta_est`,
records the fit, and checks the returned values against the original experiment
equations. Install it with `operange[idaes]`; running a fit also requires an
**Ipopt executable** on PATH. Importing the API and reading its JSON results
require only Operange's base dependencies.

## Declare experiments and fit shared parameters

Each named upstream `Experiment` supplies a root `ConcreteModel` with three
Pyomo suffixes:

- `experiment_outputs`: scalar measured Vars, mutable Params or Expressions
  mapped to observed numbers in their native physical units.
- `measurement_error`: those same components mapped to positive measurement
  standard deviations, in the same units. Required by the default weighted SSE.
- `unknown_parameters`: the shared scalar Vars or mutable Params to estimate,
  conventionally mapped to their `ComponentUID`s. An empty suffix requests
  reconciliation with known parameters.

For example, estimate one boiler efficiency from two known fuel flows:

```python
import pyomo.environ as pyo
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.core.base.componentuid import ComponentUID
from operange.integrations.parmest import ParmestCalibration, CalibrationResult

def experiment(fuel, steam, sigma):
    m = pyo.ConcreteModel()
    m.efficiency = pyo.Var(initialize=0.8, bounds=(0.1, 1))
    m.efficiency.fix()
    m.fuel = pyo.Var(initialize=fuel, units=pyo.units.MW)
    m.fuel.fix()
    m.steam = pyo.Var(initialize=0.8 * fuel, units=pyo.units.MW)
    m.balance = pyo.Constraint(expr=m.steam == m.efficiency * m.fuel)
    m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.experiment_outputs[m.steam] = steam
    m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.measurement_error[m.steam] = sigma
    m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.unknown_parameters[m.efficiency] = ComponentUID(m.efficiency)
    return Experiment(m)

campaign = ParmestCalibration(
    name="Boiler efficiency campaign",
    experiments={"run-1": experiment(10, 7.5, 0.1),
                 "run-2": experiment(20, 14.7, 0.2)},
    solver_options={"tol": 1e-10, "bound_relax_factor": 0},
    provenance={"data": "Synthetic readings; fuel flows treated as known"},
)
fit = campaign.fit()
if fit.execution != "completed":
    raise RuntimeError(fit.diagnostics)
print(fit.estimates[0].value)  # approximately 0.7425
restored = CalibrationResult.from_json(fit.to_json())
```

Each builder runs once when the campaign is constructed. Operange snapshots
the returned model; subsequent source edits do not change that campaign.
Initialize nonlinear experiments in `get_labeled_model()` before returning
them. Invalid declarations and builder/initialization errors raise during
construction. Once constructed, a missing solver, failed solve or invalid
returned candidate produces an `unresolved` result with diagnostics and no
applicable estimates.

Every experiment must label the same shared parameter paths and native units.
Individual indexed Var elements are supported. Upstream parmest cannot convert
individual indexed Param elements into unknowns; declare scalar Params or Vars
for those estimated quantities. This restriction does not apply to application
targets, which may be individual indexed mutable Param elements.
Measured outputs and recorded conditions may differ. Only explicitly labelled
unknowns are unfixed by parmest; other fixed variables remain experimental
conditions. For reconciliation, leave measured variables free and the known
equipment parameters fixed. The [boiler example](idaes.md#reuse-upstream-reconciliation-and-estimation)
demonstrates both workflows through this API.

All active experiment constraints are retained. **Remove operating service
targets from the experiment builder when they do not describe the experiment.**
A calibration at recorded inlet conditions should not invent a different
utility dispatch to meet a later operating commitment. Source objectives are
replaced by the selected least-squares objective.

`objective="SSE_weighted"` delegates to parmest's weighted objective: in the
supported API this is half the sum of squared standardized residuals per
experiment, averaged across experiments. `objective="SSE"` is also supported,
but requires one common native observation unit throughout the campaign.
This slice does not accept custom objectives, correlated measurement covariance,
regularization, experiment-specific unknown parameters or discrete models.

## Inspect and preserve the evidence

`CalibrationResult` retains:

- Model manifests and fingerprints, named measurements, units and errors,
  initial parameter values and bounds, backend versions and solver options.
- Fitted parameters, reconciled variable values, and each observation's
  `measured`, `fitted`, `residual` and `standardized_residual` values. Residual
  properties are computed from the stored measurements and fitted values.
- Constraint and bound checks in native physical units. Constraint acceptance
  uses the declared absolute `tolerance` (default `1e-5`); bounds and unchanged
  fixed conditions have zero tolerance. The objective is also recomputed from
  the returned values using the upstream objective function.
- The public API outcome, checked objective and diagnostics. Parmest's public
  return tuple does not expose raw solver termination, so `solver.termination`
  is explicitly null. A normal API return and passing source checks are recorded
  separately; they do not establish global optimality.

`calibration_result/v1` JSON includes a content-derived result identity.
`CalibrationResult.from_json()` checks its integrity and record consistency
without loading Pyomo, IDAES or pandas. It does not execute the source model or
independently verify the fit. These records are separate from operating
`process_result/v1` reports and are not inputs to `verify_result()`.

## Apply a fit explicitly, then repeat the operating study

For an existing `IdaesAdapter` with a fixed boiler efficiency parameter:

```python
calibrated = fit.apply_to(
    adapter,
    parameter_map={"efficiency": "fs.boiler_a.efficiency"},
)
before = adapter.as_claim(cases).audit_result()
after = calibrated.as_claim(cases).audit_result()
```

The map must cover every fitted parameter exactly once. Targets must be fixed
Vars or mutable real-valued Params in active model blocks, with matching native
units and compatible bounds. Inputs varied by scenarios and actuator controls
cannot be calibration targets. No unit conversion is inferred.

Application creates a fresh adapter snapshot, retaining the original adapter,
domain and requirements. Its provenance records the calibration result, study,
parent contract, explicit map, and old/new values. Calibration applies point
estimates only: the result explicitly records identifiability as `not_assessed`
and parameter uncertainty as `not_computed`. Do not interpret measurement sigma
as a bound on fitted parameters or convert a confidence region silently into a
hard operating envelope.

## Heat-exchanger calibration and re-audit

```bash
uv run --extra idaes python examples/idaes_heat_exchanger.py --calibrate \
  --calibration-output fit.json --output calibrated-study.json
```

Four synthetic experiments measure both outlet temperatures with sigma 0.1 K.
Inlet conditions, composition, **area (1 m²) and fouling factor (1)** are known.
Only the clean heat-transfer coefficient is estimated; fitting both area and
coefficient would confound their product. The noisy readings were generated
from a coefficient of 85 W/(m² K); the fit is approximately **85.00986 W/(m² K)**.

The operating model uses `U = clean_coefficient * fouling_factor`, so scenario
fouling remains distinct from the fitted equipment parameter. The same six
scenarios and original service commitments are re-audited. The original
100 W/(m² K) model has a checked feasible normal case; at the fitted coefficient,
that 4.5 kW case is unresolved. Both audits remain inconclusive, and no nonlinear
infeasibility claim is made. The original single-liquid property assumptions
and [heat-exchanger study limits](idaes.md#heat-exchanger-operating-study) apply.

The implementation follows the upstream
[parmest Experiment interface](https://pyomo.readthedocs.io/en/stable/explanation/analysis/parmest/driver.html)
and [data-reconciliation workflow](https://pyomo.readthedocs.io/en/stable/explanation/analysis/parmest/datarec.html).
For joint empirical parameter uncertainty, use the
[bootstrap and propagation guide](joint-calibration-uncertainty.md). For local
parameter information, weak combinations and held-out predictions, use the
[calibration credibility guide](calibration-credibility.md). Structural
identifiability proofs and continuous confidence-region coverage remain outside
the supported scope.
