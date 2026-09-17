# Propagate joint calibration uncertainty

The [parmest integration](calibration.md) can now bootstrap whole experiments
and carry each resulting **joint parameter vector** into finite operating
studies. Efficiency and heat loss, for example, can vary together without
combining unrelated marginal extremes into artificial equipment models.

This is empirical, finite-sample propagation. It does not create a confidence
region, a deterministic uncertainty envelope or a failure probability.

## Obtain checked joint draws

Starting from an existing `ParmestCalibration` campaign:

```python
uncertainty = campaign.bootstrap(samples=200, seed=2026)
if uncertainty.execution != "completed":
    raise RuntimeError(uncertainty.diagnostics)

joint_parameters = uncertainty.as_domain()  # ordinary FiniteSet
moments = uncertainty.statistics
print(moments["parameters"], moments["covariance"])
```

The seed and number of draws are explicit. Operange first fits the original
campaign, then calls upstream `Estimator.theta_est_bootstrap` with replacement,
the original number of experiments per draw, and `return_samples=True`.
Experiment names and repeated selections are retained. Measurements within an
experiment stay together; this API does not resample sensors independently or
add Gaussian measurement noise.

Treating the declared experiments as exchangeable sampling units is a caller
assumption. Time-dependent campaigns, grouped batches, fixed experimental
designs and heteroscedastic data may call for a different resampling method.
The current API follows parmest's experiment bootstrap; it does not supply
block, residual or parametric bootstraps.

The upstream resampler requires **more distinct experiments in each draw than
estimated parameters**. Consequently the campaign must have more experiments
than shared unknowns, and the resulting draws are conditioned on that upstream
restriction. Sampling can still fail when few distinct resamples are possible.
Operange reports that failure explicitly. The
[upstream implementation](https://pyomo.readthedocs.io/en/stable/_modules/pyomo/contrib/parmest/parmest.html#Estimator.theta_est_bootstrap)
defines this sampling behavior; it is not an unconstrained resampling guarantee.

The public bootstrap API returns parameter vectors and resampling indices,
without per-draw state values, objective values or raw solver termination.
Operange therefore fixes each vector in fresh copies of the corresponding
resampled experiments and solves for their states. The existing calibration
checker verifies the resulting equations, bounds, fixed conditions and
least-squares objective. These are **physical replays at the returned parameters**;
they do not recover or certify the original bootstrap solver's internal state
or global optimum. Initialization uses the campaign's captured source guesses.

An unsuccessful initial fit or upstream batch produces an unresolved result.
If an individual vector is invalid or its replay fails, every returned draw is
retained and the overall result is unresolved. `as_domain()`, `statistics` and
`as_claim()` require all requested draws to be checked. They never discard
unsuccessful draws and silently analyze only the survivors.

Upstream bootstrap uses NumPy's legacy global RNG. Calls through this integration
are serialized and restore its prior state on success or failure; unrelated
concurrent users of that global RNG are outside that synchronization.

## Combine equipment uncertainty with operating cases

Map fitted parameter paths explicitly to an existing `IdaesAdapter`. Declare
finite operating cases in that adapter's original input space, then supply
fixed commands for every actuator:

```python
claim = uncertainty.as_claim(
    adapter,
    operating_cases,
    parameter_map={
        "fs.boiler.efficiency": "fs.boiler.efficiency",
        "fs.boiler.heat_loss": "fs.boiler.heat_loss",
    },
    fixed_controls={"fuel": 13.5},
)
audit = claim.audit_result()
```

The original adapter stays unchanged. A fresh snapshot promotes the mapped
parameters into inputs named `parameter:<source ComponentUID>`. Native units
must match, every estimated parameter must be mapped, and every complete vector
must satisfy the target parameter bounds. Scenario inputs and actuators cannot
be targets. Scalar fixed Vars and mutable real-valued Params, including
individual indexed target elements, are supported.

The finite domain crosses **each whole vector with each operating case**.
Parameter coordinates are never recombined independently. This Cartesian stress
test assumes every such combination should be examined; it assigns no operating
case probabilities and asserts no statistical independence between the factors.
Conditional draw/case pairings are outside this factory's current scope.

Uncertain equipment parameters are not automatically observable. Omitting
actuator commands raises an error. To intentionally study ideal dispatch that
knows the actual parameter vector, use `fully_observed=True` instead of
`fixed_controls`. That choice is recorded in provenance. It is not a deployable
controller for latent parameters, or a common robust dispatch chosen before
those parameters are known.

The resulting claim uses the existing IDAES checks and result format. A `pass`
covers only the listed combinations. Failed nonlinear solves remain `unknown`,
and an incomplete operating audit remains `inconclusive`. Neither the proportion
of feasible draws nor the proportion of unresolved solves is a plant reliability
estimate. No continuous hull or ellipsoid is inferred from successful points.

## Save and inspect the evidence

```python
from operange.integrations.parmest import CalibrationBootstrap

saved = uncertainty.to_json()
restored = CalibrationBootstrap.from_json(saved)
```

`calibration_bootstrap/v1` includes the original calibration, seed, requested
count, complete parameter vectors, resampled experiment identities, physical
replays, diagnostics and scope qualifications. Its reader checks content identity
and record consistency without loading IDAES, Pyomo or pandas. It can also
construct the joint finite domain and compute empirical moments in a base-only
installation. Reading evidence is not independent mathematical verification.

`statistics` reports parameter order, native units, empirical means and the full
sample covariance matrix (divisor `n - 1`). Covariance entry `(i, j)` has the
product of the corresponding units. Repeated draws retain their multiplicity;
singular covariance is retained without inventing an inverse or confidence
ellipse. Identifiability remains unassessed.

Operating provenance links the uncertainty, original calibration, parent model,
operating cases, parameter map, dispatch assumption and scenario labels. When
counting draws, use `audit.payload.coverage.evaluated_support["scenarios"]`:
the audit may store one supporting evaluation for several identical scenarios.

## Worked boiler study

```bash
uv run --extra idaes python examples/idaes_uncertainty.py \
  --samples 12 --seed 7 \
  --uncertainty-output joint-fit.json --audit-output joint-audit.json
```

Six synthetic readings identify boiler efficiency and constant heat loss in
`steam = efficiency * fuel - heat_loss`. The joint draws are strongly positively
correlated: greater fitted efficiency accompanies greater fitted loss. Twelve
draws crossed with two demand commitments produce 24 labelled scenarios at a
fixed 13.5 MW fuel command. Near-limit cases include unresolved operating solves,
so the audit is inconclusive. The model, units and services remain explicit;
the example does not assess physical model discrepancy or plant reliability.
