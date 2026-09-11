# Quadratic response over an ellipsoid: residopt experiment

Implemented 2026-09-11 in the opt-in `operange.experimental`
namespace. The existing `process_result/v1` contract accommodates this nonlinear
case. Both backends agree on the analytical cases and the process example;
the native trust-region calculation is substantially faster on this workload.
Keep residopt as an experimental option. These results do not justify making
it a default dependency or replacing the native checker.

## Scope and interface

The [adapter](../src/operange/experimental/quadratic.py)
accepts one caller-declared quadratic response, one upper-limit requirement,
one positive-radius `EllipsoidSet`, and fixed operation with no control changes.
It provides point evaluation and a full-domain requirement audit. Sensitivity,
closest-breaking searches, recourse, intersections, and zero-radius domains
are explicitly unsupported in this experiment. It does not fit a surrogate
or establish a surrogate-error or physical-validity bound.

```python
from operange.experimental.quadratic import QuadraticProcessAdapter

# model is a QuadraticModel; domain is an EllipsoidSet with aligned coordinates.
claim = QuadraticProcessAdapter(model, backend="residopt_sdp").as_claim(domain)
evaluation = claim.evaluate_result(domain.space.nominal)
audit = claim.audit_result()
document = audit.to_json()

# The independent alternative needs neither residopt nor CVXPY.
native = QuadraticProcessAdapter(model, backend="trust_region").as_claim(domain)
```

The [executable example](../examples/quadratic_residopt.py) constructs
the model and domain. Results bundle the physical model, coordinate scales,
declared domain, numerical policy, checked evaluations, and bound evidence.
Existing result loaders read these bundles without loading the adapter or solver.
The adapter is deliberately absent from the package's top-level exports.
Benchmarking, timing aggregation, and output formatting live in the example.

## Coordinates and evidence

For model coordinates `z_i = (u_i - nominal_i) / scale_i`, the response is

```text
f(u) = offset + linear.T z + 0.5 z.T H z
residual(u) = f(u) - limit
```

Every coefficient is named and carries the response unit; the full Hessian
must be symmetric. The factor `0.5` is part of the convention. Domain and
model coordinates may have different centers, scales, and ordering, with
matching names and units. The adapter transforms the domain into a unit ball:

```text
u = domain_nominal + diag(domain_scales) radius chol(shape) y,  ||y|| <= 1
residual(u) / residual_scale = constant + b.T y + 0.5 y.T A y
```

`residual_scale` is a positive value in the response unit. Solver tolerances
apply to the resulting dimensionless bound; requirement tolerance remains
in the response unit. The physical witness is evaluated again, checked for
domain membership, and checked against the transformed expression. Numerical
boundary rounding may require moving the candidate slightly into the domain.
This affects the feasible lower bound, not the domain covered by the upper bound.

The two implementations deliberately use different numerical paths:

- The [independent implementation](../src/operange/experimental/_quadratic_bounds.py)
  caches an eigendecomposition and uses SciPy's Brent root solver for the scalar
  trust-region KKT condition. It handles concave interior maxima and the singular
  hard case. A completion-of-squares upper bound includes stationarity residual,
  estimated curvature error, and a floating-point guard. A root iteration limit
  can leave a loose valid bound; it does not imply an optimum was found.
- The [residopt implementation](../src/operange/experimental/_residopt_quadratic.py)
  forces exact SDP compilation of `QuadraticEllipsoidAtom`. For a fixed `A`,
  a CVXPY parameter supplies each new `b`. It calls the verified `solve()` API,
  requires verification of the original atom, and checks that the model remains
  fully compiled with exact original-model certificates. It then rebuilds the
  LMI using the requested coefficients and applies an eigenvalue correction.

For the SDP, `lambda >= 0` and the intended LMI is

```text
M = [lambda I - 0.5 A    -0.5 b]
    [    -0.5 b.T       t - lambda]  >= 0.
```

If the computed minimum eigenvalue is `-epsilon`, adding `2 epsilon` to
the objective bound covers `||[y; 1]||² <= 2`. The implementation also adds
rounding guards, clamps a slightly negative multiplier to zero, and rebuilds
the matrix after that change. It does not assume that an approximately
satisfied coefficient-equality constraint supplied the requested `b` exactly.

An exact rewrite certificate alone cannot establish a passing process claim.
In the SDP experiment, **both** the independently checked upper bound and the
corrected SDP bound must support a pass. A bound below an independently attained
value is rejected. A verified domain member violating the requirement proves
failure under the declared fixed policy, including when the SDP solve fails.
Without either kind of evidence, the result is inconclusive. Missing optional
dependencies or an incompatible API produce an unsupported result.

These are numerical bounds for a declared quadratic, not interval-arithmetic
proofs. Extremely scaled floating-point data remain outside the demonstrated
test envelope. No probability, model accuracy, or plant-safety conclusion is
inferred from a successful solve.

## Source and reproduction

The experiment used the sibling residopt checkout at
`5abb329600578b9782f37862db08a39fe426d5ae`. The aggregate SHA-256 of its imported
top-level Python sources was
`d3e0049ca0818e89f761265da33faa5753a816a44a6b924c85fe263474b43a89`.
Source location and hash are included in each SDP result. Installed distribution
metadata still says `0.1.0`; that label alone does not identify the tested code.

This checkout improves on the earlier installed API: `solve()` verifies original
robust atoms and handles deferred oracles, `solve_master()` explicitly does not
verify them, and `solve_report` distinguishes robust verification from master
termination. Scikit-learn is a benchmark extra in the sibling, not a runtime
dependency. The adapter checks for the current verified-solve API and requires
Python 3.13+ only when using residopt.

From the repository root, with the workspace installed and the current sibling's
runtime requirements available (including CVXPY 1.9.2 and an SDP solver):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../residopt/src .venv/bin/python \
  examples/quadratic_residopt.py \
  --source-revision "$(git -C ../residopt rev-parse HEAD)" \
  --output data/process/quadratic-residopt-experiment.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../residopt/src .venv/bin/pytest \
  tests/test_quadratic_experiment.py -q
```

The source-revision argument is a caller-supplied label; the recorded source
hash identifies the imported files. `PYTHONPATH` selects source without replacing
the installed distribution; disabling bytecode avoids writing into the sibling.
The report directory is gitignored. No package dependency or lockfile changes
are needed for this checkout-based experiment. Native tests run without residopt;
current-API integration tests skip when that optional backend is unavailable.

## Measurements

The synthetic pressure-loss example uses flow nominal 100 kg/s, scale 10 kg/s,
temperature nominal 50 °C, scale 5 °C, shape correlation 0.4 and radius 1.5.
The model has offset 25 kPa, linear coefficients `(3, -1)` kPa and Hessian
`[[4, 1], [1, 2]]` kPa. The worst attainable response is approximately
**34.88075987 kPa**. Both methods pass the 50 kPa limit and produce a verified
violating member for the 27 kPa limit. Requirement tolerance is `1e-5 kPa`.

The microbenchmark uses seeded, dense indefinite Hessians with eigenvalues
from -2 to 4 and 12 changing linear vectors per workload. Each configuration
runs three times, alternating backend order. All 432 timed calls completed
without rejected bounds or solve failures. The largest SDP upper-bound gap
above the independent feasible value was `2.14e-8`, within the requested
`1e-6` tolerance; native gaps were at most `2.49e-12`.

Median elapsed **milliseconds per 12-solve workload**, including construction:

| Dimension | Native, fresh each solve | Native, reused | SDP, fresh each solve | SDP, reused |
| --- | ---: | ---: | ---: | ---: |
| 2 | 1.117 | 0.623 | 65.233 | 15.105 |
| 8 | 1.406 | 0.660 | 73.502 | 24.618 |
| 16 | 1.781 | 0.662 | 125.296 | 58.635 |

Reuse caches the native spectrum or the residopt/CVXPY model. The first solve
includes canonicalization; later solves benefit from parameter reuse and the
solver's defaults. Residopt timing includes its original-atom verification and
the adapter's LMI correction. Imports, common independent reference calculations,
physical normalization, and result serialization are outside these timings.
This measures backend cost, not end-to-end process adapter latency. Strategy
selection, multiple constraints, changing Hessians, and design optimization
were not measured. Three short repetitions on one machine give a directional
comparison, not a cross-platform performance claim.

Environment: Python 3.13.1, NumPy 2.2.6, SciPy 1.15.3, CVXPY 1.9.2,
CLARABEL, macOS 26.5.2 arm64, Apple Accelerate BLAS. The run recorded 366
NumPy `matmul` divide/overflow/invalid warnings across the fixture construction,
native spectral checks, and residopt atom construction. Similar warnings were
reproduced in a separate NumPy-only QR/matrix-product calculation. All resulting
fixtures were finite and their spectra agreed with the intended eigenvalues to
within `2.3e-15`; the independent analytical and bound checks passed. The cause
has not been established. The JSON retains warning categories, locations, and
counts, and warning-recording overhead is included in timing. Recheck these
measurements on another numerical-library build before treating them as a
performance baseline.

## Decision and next experiment

For a fixed quadratic over one ellipsoid, use the native checker. Reusing the
SDP model improves its own workload time by roughly 2–4 times, but it remains
about 24–89 times slower than native reuse in this run.

The stronger potential role for residopt is embedding robust process constraints
inside a **coupled engineering design optimization**, where its compiler and
master/oracle machinery replace work the process package otherwise has to do.
A useful next bounded experiment would optimize a small number of engineering
design variables subject to several declared process constraints, keeping the
design dependence within supported atom forms, and independently re-audit the
returned design. That is a separate decision; this change does not implement
it or add general optimizer abstractions to the DSL.

Follow-up: the [coupled design experiment](coupled-design-experiment.md) is now
implemented in consumer example code, comparing native exchange, direct SDP,
fully compiled residopt, and hybrid residopt with independent final audits and
objective lower bounds.
