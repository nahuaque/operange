# First example: a heat-recovery capacity claim

Part of the [process exploration](README.md). All equipment values and costs
below are synthetic. The results are analytical reference calculations for
this deliberately simplified model. The [first prototype](../README.md)
now reproduces the static example numerically and checks against these references.

The report and CAPEX ranking are prototype conveniences. Under the target
[package boundary](package-boundary.md), the core supplies engineering results
and consumers own valuation, selection, and presentation.

## Fixed design and physical equations

A heat pump supplies a heat network using recovered source heat. Use a fixed
COP of `C = 4` within this toy operating envelope. The model has no storage,
temperature dependence, startup, or transient dynamics yet.

| Quantity | Value | Meaning |
| --- | --- | --- |
| `S` | 6 MW thermal | Installed source-heat capture capacity before derating |
| `P` | 2 MW electrical | Compressor input-power capacity |
| `H0` | 6 MW thermal | Baseline heat delivery requirement |
| `f` | 0 to 0.25 | Fractional loss of source-heat capacity |
| `delta` | 0 to 0.25 | Fractional increase in required delivered heat |

`f` is a capacity derating, not a fouling resistance. Mapping physical fouling
resistance or exchanger area to capacity would require another model.

The uncertainty set is the full rectangle in `(f, delta)`. It has no probability
interpretation. The nominal realization is `(0, 0)`. The operator observes both
inputs before choosing electrical power `p` and delivered heat `q`.

The physical constraints and claim are

$$
0\le p\le P,\qquad q=Cp,\qquad q_s=q-p,
\qquad 0\le q_s\le S(1-f),
$$

$$
q\ge H_0(1+\delta).
$$

Both electrical energy and source heat contribute to delivery. Treating `S`
as delivered heat would change the model and the numerical results.

Eliminating the controls gives the maximum recoverable delivery:

$$
q_{\max}(f)=\min\left(CP,\frac{C}{C-1}S(1-f)\right)
           =8(1-f)\ \mathrm{MW}
$$

for the baseline design over the declared domain. Every value between zero
and this maximum is feasible. Thus the claim is feasible exactly when

$$
8(1-f)\ge6(1+\delta).
$$

This calculation resolves all allowed static recourse. It does not infer
failure from an unsuccessful attempt to find operating settings.

## A closest breaking realization

Define severity using the declared full-scale disturbances:

$$
D(f,\delta)=\max(f/0.25,\delta/0.25).
$$

Within radius `r` for `0 <= r <= 1`, the smallest delivery slack occurs at
`f = delta = 0.25r` because the slack decreases in both coordinates:

$$
q_{\max}-H_0(1+\delta)=2-3.5r\ \mathrm{MW}.
$$

The feasibility boundary is therefore `r = 4/7`, approximately `0.571429`.
Equality still satisfies the claim. Strictly failing points approach that
radius from above, so the strict-failure problem has an infimum there, rather
than an attained closest failure.

For an attained witness, require a delivery shortfall of at least `0.01 MW`
(10 kW). Then

$$
r_{0.01}=\frac{2+0.01}{3.5}=\frac{201}{350}\approx0.574286.
$$

| Witness field | Reference value |
| --- | --- |
| Source-capacity derating `f` | 14.357143% |
| Demand increase `delta` | 14.357143% |
| Maximum delivery | 6.851429 MW thermal |
| Required delivery | 6.861429 MW thermal |
| Unrecoverable shortfall | 0.010000 MW thermal |
| Best available compressor input | 1.712857 MW electrical |
| Binding physical limit | Available source heat |
| Unused compressor capacity | 0.287143 MW electrical |
| Evidence | Analytical global minimum for this domain, distance, and shortfall margin |

The witness is admissible, and the formula for `q_max` proves that no allowed
operating adjustment restores delivery. The monotonic slack bound excludes
every closer margin-separated failure.

A fixed-power comparison illustrates why recourse matters. Holding the nominal
`p = 1.5 MW` freezes delivery at 6 MW: any positive demand increase fails. For
the same 10 kW separation, `delta = 1/600` and `f = 0` gives a minimum severity
of `1/150`, approximately `0.006667`. The adjustable plant supports much larger
disturbances than this fixed-operation contract.

## Repair the full uncertainty set

At the worst point `(f, delta) = (0.25, 0.25)`, required delivery is 7.5 MW.
The same monotonicity argument makes this corner sufficient for this particular
model. A generic nonlinear model does not inherit that shortcut.

| Candidate | `S` (MW thermal) | `P` (MW electrical) | Illustrative added CAPEX | Worst delivery | Minimum slack | Original claim |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 6 | 2 | £0 | 6 MW | -1.5 MW | Fail |
| Compressor uprate | 6 | 2.5 | £100,000 | 6 MW | -1.5 MW | Fail |
| Small source-capture upgrade | 7 | 2 | £150,000 | 7 MW | -0.5 MW | Fail |
| Source-capture upgrade | 7.5 | 2 | £240,000 | 7.5 MW | 0 MW | Pass at boundary |
| Larger source-capture upgrade | 8 | 2 | £310,000 | 8 MW | +0.5 MW | Pass |

The £240,000 candidate is the cheapest passing option in this synthetic
catalog. It has zero worst-case headroom. Requiring a positive operating
reserve would be an explicit additional requirement and would change the
selection. These costs do not estimate a real plant or establish economic
optimality beyond the catalog.

The source upgrade changes `S` while retaining the original fractional derating
domain; that invariance is part of the toy assumptions. Whether a real upgrade
changes fouling behavior would need a new physical treatment.

Reducing `H0` to 4.8 MW also makes the baseline design feasible throughout the
domain. That is a lower service commitment, and should be reported as a revised
claim rather than a repair that preserves the original promise.

## Sketch of the eventual user experience

```python
# Proposed workflow only. None of these process functions exists yet.
claim = process_claim(
    name="Meet heat demand throughout the declared operating envelope",
    design=baseline_design,
    model=heat_recovery_model,
    uncertainty=derating_and_demand_box,
    recourse=fully_observed_static_controls,
    requirement=heat_delivery_requirement,
)

audit = claim.audit()
witness = claim.breaking_witness(
    distance=normalized_max_excursion,
    shortfall_margin={"heat_delivery": "0.01 MW"},
)
repairs = claim.compare_repairs(repair_catalog)
contract = repairs.selected_passing_candidate.freeze()
```

The report should expose the equations or model reference behind every named
object, then show the verdict, witness, recoverable operation, repair table,
and evidence scope. Freezing does not upgrade the evidence level.

## Second experiment: add storage and causal dispatch

Add hourly source availability, demand, electricity prices, and thermal storage.
With charging/discharging powers in MW and stored energy in MWh, use

$$
E_{t+1}=(1-\lambda_t\Delta t)E_t
       +\eta_c c_t\Delta t-\frac{b_t}{\eta_d}\Delta t,
$$

$$
q_t+b_t-c_t\ge H_t,\qquad 0\le E_t\le E_{\max},
$$

with declared power bounds and `0 <= lambda_t * dt <= 1`. Specify any prohibition
on simultaneous charging/discharging. Fix initial storage and a terminal rule
so a dispatch cannot claim sustainable operation merely by emptying the store
at the end of every evaluation window.

Begin with a two-stage tree whose futures share the same first-stage
observations. Force shared first-stage actions and compare against separately
optimized perfect-foresight paths. Include a case where those pathwise solutions
all succeed but no common first-stage action succeeds in every branch. Such a
case may need both branches as its breaking artifact.

The dynamic prototype should answer a specific question: how do the witness
and selected repair change when the controller learns about a heat-demand surge
only after its storage charging decision? Keep model coefficients simple enough
that a small tree model supplies independent reference evidence.
