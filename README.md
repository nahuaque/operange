# Operange

**Check whether a process can keep meeting its requirements as operating
conditions change, and identify what limits its flexibility.**

`operange` is a Python library for process engineers studying
uncertain loads, feed conditions, equipment capacities and operating decisions.
It provides a small engineering language for specifying a robustness study and
returning process values, sensitivities, failing conditions and the evidence
behind a verdict.

**Status:** alpha, developed as an independent project.
The implemented models are deliberately bounded; current capabilities are
listed below. The distribution is independent of `updatesupport` and requires
Python 3.10+, NumPy and SciPy. The development version is `0.1.0`.

## Install from this checkout

```bash
python -m pip install .
```

Or build a wheel from the repository root and install it:

```bash
uv build --wheel --out-dir dist
python -m pip install dist/operange-0.1.0-py3-none-any.whl
```

## Try the shared-steam example

```python
from operange import (
    AffineOutput, AffineProcessAdapter, AffineRequirement, AffineTerm,
    BoxSet, BudgetSet, Parameter,
)

loads = BoxSet((
    Parameter("dryer", "MW", 10, 8, 12, 2, "Synthetic dryer envelope"),
    Parameter("evaporator", "MW", 6, 4, 8, 2, "Synthetic evaporator envelope"),
))
model = AffineProcessAdapter(
    name="Shared steam header",
    input_space=loads.space,
    outputs=(AffineOutput(
        "steam", "MW", "thermal_power", 0,
        tuple(AffineTerm(n, 1, "MW/MW") for n in loads.space.names),
        "Sum of process steam loads",
    ),),
    requirements=(AffineRequirement("header_capacity", "steam", 19),),
)
claim = model.as_claim(loads)
print(claim.evaluate_result(loads.nominal).to_json())
print(claim.sensitivity_result(loads.nominal).to_json())
print(claim.audit_result().payload.verdict)  # fail: peak demand is 20 MW
print(model.as_claim(BudgetSet(loads, 1)).audit_result().payload.verdict)  # pass
```

Nominal steam demand is 16 MW. The box permits both users to peak at once;
the budget allows only one full normalized deviation in total. Changing that
assumption creates a different claim, not an equipment repair.

The [getting-started guide](docs/index.md)
extends this model with a shared boiler, evidence inspection, an equipment
change and JSON round trips. The complete consumer example is
[`examples/steam_header.py`](examples/steam_header.py).

## What works today

| Model | Implemented questions | Scope |
| --- | --- | --- |
| Caller-declared affine model: constant plus weighted inputs and fixed controls | Point evaluation, analytical first sensitivities, requirement audits and boundary/positive-violation distances | Audits over supported uncertainty domains; distance searches over boxes and polytopes with explicit normalized L-infinity scales; controls remain fixed |
| Caller-declared linear model with bounded controls | Joint dispatch feasibility, finite-scenario audits and checked infeasibility witnesses | Fixed or fully observed single-stage operation; coupled equipment limits always apply; no dispatch derivatives or distance searches |
| Heat-recovery reference model | Heat-delivery evaluation, optimized-response sensitivities, full-box audits, feasibility boundary, positive-shortfall breaking search and engineering-change comparison | Static, constant COP, two uncertain inputs and fully observed electrical-power recourse |
| Thermal-storage reference model | Fixed, causal and perfect-foresight audits; conflicting futures; equipment and earlier-information changes | Complete declared finite two-period tree; no general trajectory solver or sensitivity operator |
| Supplied startup profiles | Peak and integrated load, shared-capacity audits and comparison of fixed start schedules | Finite amplitude/duration/timing scenarios; all time segments of the declared piecewise-linear profiles; no motor dynamics or adaptive scheduler |
| Heat-cascade reference model | Minimum heating/cooling targets, maximum heat recovery, pinch locations and utility-target audits | Finite steady-state sensible-heat scenarios at fixed ΔTmin; no installed exchanger-network feasibility or sensitivity operator |

The [linear dispatch guide](docs/linear-dispatch.md) adds two adjustable boilers
and a shared fuel supply. Redispatch handles either individual process-load peak;
the combined peak remains infeasible. Increasing the shared fuel capacity passes
all four declared scenarios. Each pass includes checked physical controls, and
each failure includes a constraint combination that rules out all permitted
controls. Solver termination alone cannot establish either conclusion.

The [failure-distance guide](docs/failure-distance.md) finds which fixed affine
requirement first reaches its limit. The shared steam header reaches its boundary
at normalized distance 0.75; a 0.01 MW violation requires distance 0.7525.
Distances carry verified lower and upper bounds, and an unresolved branch cannot
silently disappear from the search for the nearest failure.

The [startup example](docs/startup.md)
keeps a compressor start distinct from ordinary load variation. Two synthetic
starts reach 700 kVA across the 65 declared scenarios; staggering the commands
reduces that maximum to 480 kVA against a 500 kVA limit. The uncertainty and
equipment remain unchanged. `PiecewiseLinearProfile` makes interpolation and
the running tail explicit, so the audit checks between supplied knots as well
as at them. Coverage does not extend to unmeasured waveform spikes or uncertain
values between the finite scenarios.

The [heat-cascade example](docs/pinch.md)
screens heat-integration targets across four steady operating cases. Nominal
heating needs 10 kW; reduced hot-stream flow or increased cold-stream flow each
raise it to 40 kW, while both together require 100 kW and exceed a 50 kW limit.
The pinch location changes between scenarios. A passing target screen does not
prove an installed exchanger network works: connections, area, fouling and
controls require a network model. Heat-cascade data are returned for downstream
use without plotting or economics in the package.

Uncertainty declarations include independent bounds (`BoxSet`), explicit finite
cases, fractions summing to one (`SimplexSet`), budgets limiting simultaneous
deviations, ellipsoids and linear coupled restrictions (`PolytopeSet`). Named
vector inputs and domain composition are also available. Representing a domain
does not automatically give every model an optimizer or a proof over it.

The intended model boundary can cover a coupled system, as the shared-utility
example does. Current implementations do not establish whole-plant scalability
or arbitrary nonlinear flowsheet robustness. External simulator adapters,
general dynamics, degradation, faults, model ensembles and distributional
ambiguity remain future work. Your model supplies the equations and consistent
units; the package does not supply thermodynamic property packages or automatic
unit conversion.

## The engineering question

Suppose a dryer and an evaporator share a steam header rated for 19 MW.
Their nominal demands are 10 MW and 6 MW, and each can vary by 2 MW.

| Operating condition | Steam demand | Within 19 MW capacity? |
| --- | --- | --- |
| Nominal production | 16 MW | Yes |
| Dryer at maximum, evaporator nominal | 18 MW | Yes |
| Evaporator at maximum, dryer nominal | 18 MW | Yes |
| Both at maximum | 20 MW | No |

Both individual demand increases fit. The combined increase does not. An audit
of the full declared load envelope finds this failing combination and identifies
the shared capacity constraint.

There are different engineering responses. A larger header can accommodate the
original load envelope. A scheduling restriction can prevent the two peaks
from coinciding, but that changes the operating commitment. Reducing required
production changes the service requirement. Keeping those distinctions explicit
is central to the package.

The same workflow applies to a supported model of heat recovery or storage:
which disturbance causes a requirement to fail, can permitted operating
adjustments recover it, and does a proposed engineering change withstand the
whole original envelope?

## What you specify, and what you get back

A study makes five things explicit:

1. **The process model and design.** Balances, capacities and calculated outputs,
   such as steam demand, heat delivery or stored energy.
2. **What can vary together.** Bounds or coupled restrictions on uncertain
   inputs, with units, nominal values and the source of those assumptions.
3. **What operators may adjust.** The allowed controls and when the information
   needed to choose them becomes available.
4. **What must remain true.** Named engineering requirements, such as meeting
   heat demand or staying within a shared utility capacity.
5. **The question to evaluate.** A particular operating point, a sensitivity,
   an audit of the declared envelope, or a supported search for a nearby failure.

The outputs are structured engineering results for your own code to consume:
physical quantities and constraint margins; derivatives with their units and
operating assumptions; pass, fail or inconclusive audit outcomes; and witnesses,
bounds and coverage explaining what was established. A **witness** is a concrete
admissible condition, or a conflicting set of futures, that demonstrates failure
under the declared operating permissions.

A failed numerical solve is kept separate from physical infeasibility. A pass
over an explicitly finite set of scenarios has that finite scope. A nearest
failure is reported as such only when the search evidence supports it. Numerical
tolerances remain part of the result.

## Public API and evidence scope

- `operange`: shared DSL, uncertainty geometry, fixed affine model
  adapter and typed `process_result/v1` results.
- `operange.reference`: bounded heat-recovery and two-period
  storage models, supplied-startup profile audits, heat-cascade target audits,
  and model-specific engineering changes and numerical records.
- `operange.experimental`: opt-in research APIs without a stability
  promise. The residopt experiment is parked and adds no runtime dependency.

See the [API and capability guide](docs/api.md)
for supported declarations, result fields and model-specific restrictions.
Unsupported operations return explicit statuses rather than an inferred pass.

## Consumer examples

From the repository root:

```bash
uv run python -m examples.steam_header
uv run python -m examples.heat_recovery
uv run python -m examples.thermal_storage
uv run python -m examples.startup
uv run python -m examples.pinch
```

These repository examples own Markdown, prices, ranking and selected-repair
logic. They are not installed with the process wheel. Core repair comparisons
return engineering audits in caller order and do not select an investment.

Engineering results use `process_result/v1`. Saved result bundles and frozen
static heat contracts are tested against the installed wheel.

## Methods and background

See [methods, scope and design lineage](docs/methods.md) for the connections to
classical flexibility analysis, robust optimization, sensitivity methods,
downstream techno-economic analysis and the original `updatesupport` extraction.
