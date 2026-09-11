# Coupled engineering design with residopt

Implemented 2026-09-11 as a consumer-owned
[example](../examples/coupled_design.py), following the
[fixed quadratic experiment](quadratic-residopt-experiment.md). No new optimizer
API, dependency, or result type was added to `operange`.

The result supports a more useful role for residopt than the first experiment:
it compiles several design-dependent robust constraints and manages deferred
constraints with its verified master/oracle loop. Fully compiled residopt agrees
with a direct SDP, with modest overhead on this small problem. The hybrid path
works but is slower here. This supports keeping it available as an optional
consumer backend; it does not establish a performance advantage for hybrid
selection or justify a default dependency.

## Engineering problem

Size an exchanger, cooler, and pump together to minimize synthetic equipment
material mass while satisfying temperature, pressure, and cooling requirements
for every member of one correlated feed-uncertainty ellipsoid. All coefficients
are synthetic local design-response equations, not a validated equipment model.
The purpose is to test compilation, optimization evidence, and package boundaries.

| Normalized decision | Physical quantity | Scale | Bounds |
| --- | --- | --- | --- |
| `a` | Exchanger area | 10 m² | 0–40 m² |
| `c` | Cooler rating | 100 kW | 0–400 kW |
| `p` | Pump rating | 10 kW | 0–40 kW |

The shared footprint index satisfies `a + 0.4 c + 0.6 p <= 4.8`.
The baseline material objective is `120 a + 90 c + 60 p` kg. Six positive
material-factor vectors exercise alternative manufacturing assumptions; these
are declared example inputs rather than package-owned valuation logic.

Uncertain normalized coordinates are feed flow around 100 kg/s with scale
10 kg/s, and feed temperature around 50 °C with scale 5 °C. Their ellipsoid has
shape `[[1, 0.35], [0.35, 1]]` and radius 1.2. The shape specifies geometry;
no probability distribution or confidence level is inferred.

For `x = (a, c, p)` and uncertain normalized feed `z`, each normalized residual is

```text
q_i(x, z) = 0.5 z.T H_i z + (S_i x + s_i).T z + r_i.T x + offset_i.
```

The residuals measure outlet-temperature excess (scale 5 K), pressure-drop
excess (scale 10 kPa), and cooling-duty shortfall (scale 100 kW), each required
to be nonpositive within its declared numerical tolerance. The example records
every coefficient and unit in the JSON report.

The equipment decisions are coupled. More exchanger area reduces temperature
excess and cooling shortfall but increases pressure demand. Cooler size also
affects pressure demand. Pump size reduces the pressure shortfall but adds
thermal burden. Design choices change both nominal residuals and sensitivity to
feed uncertainty through `S_i x`. The cooling quadratic is indefinite.

The uncertainty Hessians and ellipsoid are independent of design. For fixed
`z`, each requirement is affine in `x`, so robust feasible design sets are convex
and fit the supported `QuadraticEllipsoidAtom` form. This experiment does not
support design-dependent uncertainty geometry, arbitrary design-dependent
Hessians, simulator convergence, discrete equipment catalogs, or operator recourse.

## Four solution paths

1. **Native exchange reference.** Start with nominal scenario cuts, solve a
   SciPy/HiGHS LP, find each constraint's worst feed using the independent
   trust-region calculation, add violated scenario cuts, and repeat. Mapped
   scenarios are checked against the physical uncertainty domain before use.
2. **Direct SDP.** Build the three S-lemma LMIs directly in CVXPY. This path
   uses no residopt import. It provides a comparison with hand-written conic
   modeling, not an independent theory of the robust counterpart.
3. **Fully compiled residopt.** Supply three quadratic atoms, the shared design
   bounds, footprint constraint, and parameterized material objective to the
   compiler. Call the verified `solve()` API and require exact original-model
   certificates and checks for all three original atoms.
4. **Hybrid residopt.** Compile the temperature atom; defer pressure and cooling
   to residopt's oracle/cut loop. This is an explicit experimental policy, not
   an automatically selected or learned strategy. A verified hybrid result
   need not be fully compiled; original-atom checks and the independent final
   audits determine feasibility.

The direct and fully compiled SDP formulations are numerically the same conic
problem. Their objective values agreed within `4e-11 kg` across repeated solves.
The other methods agreed with the direct SDP to within `6.5e-6 kg`, even in a
case with a flat objective where design coordinates differed slightly.

## Feasibility and optimality are separate checks

Every returned design is frozen into three existing `QuadraticModel` objects.
The existing native `QuadraticProcessAdapter` then audits the physical ellipsoid
and emits three portable `process_result/v1` bundles. These checks do not call
residopt or depend on its oracle results. Box bounds and the shared footprint
are checked separately. The consumer report retains the physical equipment
sizes, objective, design constraints, optimization evidence, and process bundles.
Each process bundle freezes the resulting response coefficients; the enclosing
example report links them to the original design equations and equipment sizes.

A solver's optimal termination does not establish global optimality in this
report. The independent scenario-cut LP supplies a numerical lower bound on
the objective over the declared feasible design set. For cuts `A x <= b` and
any nonnegative multiplier vector `mu`,

```text
lower = -mu.T b + sum_j min((w + A.T mu)_j lower_j,
                            (w + A.T mu)_j upper_j).
```

This is the infimum of the Lagrangian over the design box. The implementation
projects the LP's multiplier estimates to nonnegative values, recomputes this
bound, and subtracts a floating-point guard. It does not trust the LP objective
as an optimality certificate. The report includes the cut points, matrix, right
hand sides, box, multipliers, weights, and guard so the bound can be replayed.

The numerical policy is explicit:

- Optimization requests a normalized reserve of `1e-5` below each limit.
- Original-atom separation tolerance is `1e-7`; inner trust-region and LP
  tolerances are `1e-9`. The final physical audit allows `1e-6` times the declared
  residual scale. Design constraints use a normalized tolerance of `1e-6`.
- The objective lower bound relaxes scenario cuts and design bounds to those
  final audit tolerances. It therefore covers the same toleranced feasible set,
  rather than using the stricter reserve to overstate the lower bound.
- A design is labeled `verified_near_optimal` only if its physical audits and
  design checks pass and its material mass lies between the lower bound and
  `lower bound + 0.02 kg`. Otherwise it is unresolved. A failed physical claim
  retains its counterexample in the existing result bundle.

These are floating-point numerical certificates for the declared equations.
Model approximation error, manufacturing uncertainty, and plant validity are
not supplied by successful optimization. Exhausted iteration budgets, incomplete
verification reports, and solver failures do not confer feasibility or optimality.

## Measured outcome

With baseline material factors, the fully compiled residopt result was:

| Quantity | Nominal-only design | Robust design |
| --- | ---: | ---: |
| Exchanger area | 12.788 m² | 18.736 m² |
| Cooler rating | 169.949 kW | 185.846 kW |
| Pump rating | 19.896 kW | 29.552 kW |
| Synthetic material mass | 425.784 kg | 569.409 kg |

The nominal design meets all three equations at nominal feed, but its worst
residuals are approximately **+6.227 K**, **+16.743 kPa**, and **+53.055 kW**.
Each failing audit includes a feasible breaking witness. The robust design has
independently checked upper residuals of approximately `-0.00004995 K`,
`-0.00009995 kPa`, and `-0.00100292 kW`, so all requirements pass.

The independent material-mass lower bound is **569.407748 kg**. The robust
design's mass is **569.408896 kg**, leaving a gap of **0.001149 kg**. This is a
numerical near-optimality statement for the synthetic model and declared
tolerances, not a claim about sensible manufacturing precision.

Across six material-factor cases, three repeats, four backends, and fresh/reused
preparations, **all 144 design solves passed the independent feasibility and
objective-gap checks**. The maximum gap was `0.002339 kg`. Changing material
factors made the footprint bound active in one case and the cooler's upper size
bound active in another. Fresh native solves took 3–12 LP rounds; fresh hybrid
solves took 3–7 master rounds and added 2–7 cuts.

Median milliseconds **per six-design workload**:

| Backend | Fresh build + solve | Reused build + solve | Fresh with process audits/bundles | Reused with process audits/bundles |
| --- | ---: | ---: | ---: | ---: |
| Native exchange LP | 36.57 | 26.78 | 143.18 | 137.67 |
| Direct SDP | 51.18 | 14.69 | 165.66 | 126.92 |
| Residopt SDP | 63.83 | 21.97 | 179.44 | 135.20 |
| Residopt hybrid | 145.91 | 77.57 | 264.68 | 193.35 |

Fresh workloads rebuild before every design; reused workloads retain the fixed
equations/domain and update material objective coefficients. Native reuse retains
its spectrum and scenario cuts; hybrid reuse retains compiled constraints and
valid accumulated cuts. CVXPY uses its default solver reuse behavior. Backend
order alternates across repeats. First-solve canonicalization, residopt's own
original-atom verification, and failed-attempt time are included. Identical
physical audits and result bundling run for every method and every case.

Imports, common independent lower-bound precomputation, the nominal comparison,
and final JSON file writing are outside these timings. Native exchange also
computes its own lower bound as part of its timed LP loop. The table therefore
compares candidate generation plus common process auditing, not the total cost
of generating every independent reference from scratch. Three short repeats
on one machine are directional measurements, not a general performance claim.

The finalized run emitted no numerical warnings. Environment: Python 3.13.1,
NumPy 2.2.6, SciPy 1.15.3, CVXPY 1.9.2, CLARABEL, macOS 26.5.2 arm64.
This small two-coordinate test does not resolve the separate matrix-product
warning observed in the earlier higher-dimensional benchmark.

## Reproduce and inspect

The sibling residopt checkout remained unchanged at
`5abb329600578b9782f37862db08a39fe426d5ae`, with imported source SHA-256
`d3e0049ca0818e89f761265da33faa5753a816a44a6b924c85fe263474b43a89`.
As in the first experiment, installed distribution metadata alone does not
identify this source revision. Current residopt requires Python 3.13+.

From the repository root, using the existing workspace environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../residopt/src .venv/bin/python \
  examples/coupled_design.py \
  --source-revision "$(git -C ../residopt rev-parse HEAD)" \
  --output data/process/coupled-design-experiment.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../residopt/src .venv/bin/pytest \
  tests/test_coupled_design.py -q
```

The example's report is gitignored under `data/`. Its representative cases
include process bundles, optimizer evidence, independent objective bounds,
measurements, warning records, and source identity. The source-revision label
is caller-supplied; the recorded hash identifies the imported source files.
The [25 tests](../tests/test_coupled_design.py) cover physical mapping,
nominal failures, changed active constraints, objective-bound replay, domain
membership of cuts, reuse, inaccurate/incomplete solves, and iteration limits.
Native tests need no optional solver package; the direct path needs CVXPY;
current-residopt integration tests skip if that backend is unavailable.

## What this changes

Residopt is now demonstrated in the role that looked promising: optimizing
shared equipment decisions across several robust process constraints while
retaining evidence about the original atoms. Its fully compiled path is close
to direct SDP performance in this small workflow, with original-constraint
verification and optional cut management supplied by the library.

The native exchange method remains competitive, and hand-selected hybrid
decomposition adds overhead for these three small constraints. A larger or
mixed constraint family is needed before claiming a hybrid advantage.

Keep the current separation: downstream code chooses the engineering design
variables, objective, and optimization backend; the process package evaluates
and audits frozen designs through its existing DSL and evidence contract.
No additional general optimization or presentation abstraction is justified
by this experiment alone.
