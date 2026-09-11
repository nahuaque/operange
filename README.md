# Operange

**Check whether a process can keep meeting its requirements as operating
conditions change, and identify what limits its flexibility.**

`operange` is a Python library for process engineers studying
uncertain loads, feed conditions, equipment capacities and operating decisions.
It provides a small engineering language for specifying a robustness study and
returning process values, sensitivities, failing conditions and the evidence
behind a verdict.

Operange studies **physical process responses and feasible operation**. It
provides engineering sensitivity and robustness analysis, rather than
hidden-variable recomposition. The separate `updatesupport` project studies
how reported aggregates change with subgroup mix. The projects can contribute
to different parts of a techno-economic analysis (TEA), as described below.

**Status:** alpha, developed as an independent project.
The implemented models are deliberately bounded; current capabilities are
listed below. The distribution is independent of `updatesupport` and requires
Python 3.10+, NumPy and SciPy. The development version is `0.1.0`.

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

## Robustness, flexibility and operator adjustments

Here, **robustness** means satisfying the declared requirements throughout the
chosen uncertainty envelope, under the stated operating permissions.
**Operational flexibility** concerns the range of conditions the design can
accommodate by adjusting operation. This follows the established process-design
meaning of flexibility; Swaney and Grossmann formalized an index measuring a
feasible steady-state parameter region with adjustable controls.
([Swaney and Grossmann, 1985](https://doi.org/10.1002/aic.690310412))

The operating permissions matter as much as the equipment:

| Operating assumption | Engineering meaning | Current example |
| --- | --- | --- |
| Fixed operation | The declared control settings remain fixed while uncertain inputs vary. | Affine utility model |
| Adjustment after observation, also called recourse | Operators can choose a permitted setting after observing the current disturbance. | Static heat-recovery model |
| Causal operation | Decisions can use information already available, with earlier decisions shared between futures that have not yet become distinguishable. | Two-period thermal storage |

For storage, preparing for a demand surge may require charging, while preparing
for incoming surplus heat may require leaving space. Each future can be feasible
when planned with advance knowledge, yet there may be no preparation decision
that accommodates both before the event is known. The storage reference model
can expose that information conflict and compare it with perfect foresight.

Feasible steady-state adjustment alone does not establish that a real controller
can reach the operating point, remain stable, or satisfy constraints during a
transient. Those questions require the corresponding dynamic model and evidence.
The current storage model addresses only its declared two-period horizon.

## How this relates to familiar methods

The contribution is a reusable way to declare the engineering question, run
supported methods and preserve their evidence. It builds on established
sensitivity, flexibility and robust-optimization ideas.

| Familiar approach | What it tells you | How `operange` relates |
| --- | --- | --- |
| Nominal simulation and design margins | Whether the design works at specified conditions, with an allowance for variation. | Makes the variation envelope explicit and checks which combinations exhaust the available margin. The supplied physical model still determines the answer. |
| Parameter sweeps and one-at-a-time sensitivity | How responses change at selected points or along selected directions. | Returns named local derivatives and separately audits requirements over a supported domain. A local derivative or a sweep alone does not establish a full-domain pass. |
| Monte Carlo uncertainty analysis | Output distributions and estimated violation frequencies under assumed input distributions. | The current audits ask whether admissible conditions can violate a requirement, without assigning their probability. A set-based robustness margin is not a failure probability; no failures in a finite sample alone is not an exhaustive proof. See [Monte Carlo uncertainty analysis](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=929780). |
| Global sensitivity analysis, such as Sobol indices | How inputs and their interactions contribute to output variance under the analysis distribution. | Local derivatives and constraint-failure evidence answer different questions. The package does not currently compute Sobol indices; a tool such as [SALib](https://salib.readthedocs.io/en/latest/user_guide/basics.html) can complement it. |
| Classical flexibility analysis | Whether operating adjustments can maintain feasibility over a parameter region, and how large that region can be. | This is a direct mathematical foundation. The package adds explicit model, uncertainty and operating declarations, typed results and repeatable engineering-change audits; it does not claim to invent a new flexibility index. |
| Robust optimization | Find decisions that satisfy constraints across an uncertainty set, potentially with adjustable controls. | Uses the same concepts, with an initial focus on auditing supplied designs. General robust design optimization is not part of the supported API. Existing tools such as [PyROS](https://pyomo.readthedocs.io/en/stable/explanation/solvers/pyros/index.html) solve broader nonlinear two-stage problems; there is currently no PyROS integration. |
| Model predictive control, including robust MPC | Choose control moves repeatedly using a dynamic model, measurements and a prediction horizon. | Makes operating permissions and information timing part of an audit. It does not implement an online controller or replace a controller framework such as [do-mpc](https://www.do-mpc.com/en/latest/theory_mpc.html). |

Different tools can therefore be useful in the same study: sensitivity analysis
to understand responses, probabilistic analysis to estimate frequency, and a
robustness audit to establish the limits of a stated engineering commitment.
A narrower uncertainty envelope can make a robustness claim easier to satisfy;
its physical justification remains part of the study.

## Where techno-economic analysis fits

For a TEA, this package supplies the engineering side: physical outputs,
constraint evidence, sensitivities and the consequences of engineering changes.
Your downstream model converts those outputs into energy bills, installed costs,
annual cash flows or other economic measures. Pricing, investment selection,
financial attribution and presentation belong to that downstream code.

Hidden-variable recomposition can still be useful in the **economics reporting**
part of a TEA, using `updatesupport` and, where its financial interfaces fit,
[`updatesupport-finance`](https://github.com/nahuaque/updatesupport/blob/08b3652bb6f8e9c7437b4329632c6e6583e9b96a/packages/updatesupport-finance/README.md).
For example, an annual unit-cost summary might hide a different mix of operating
hours, load levels or tariff periods within the same reported categories. With
retained detailed rows and a suitable declared metric, a recomposition audit can
ask how sensitive that aggregate is to the hidden mix. This is a possible
application of those APIs, not a built-in plant-TEA integration.

| Part of the study | Question | Responsibility |
| --- | --- | --- |
| Process engineering | Can the design meet its requirements across the specified physical conditions and operating permissions? | `operange` and the supplied engineering model |
| Economic calculation | What are those physical outcomes worth under the chosen cost and financial assumptions? | Your TEA/economic model |
| Aggregate economic reporting | Could an unreported change in the retained operating mix alter a reported economic conclusion? | `updatesupport`, with suitable `updatesupport-finance` interfaces and caller-supplied metrics |

Reweighting precomputed economic rows does not re-solve a coupled plant or its
operating schedule. If a changed mix changes the physical response, the downstream
study must obtain consistent engineering evaluations. `updatesupport-finance`
currently focuses on financial model-risk reporting and disclosure analysis;
it is not a ready-made plant cash-flow model. Neither sibling package is a
runtime dependency of `operange`.

## What works today

| Model | Implemented questions | Scope |
| --- | --- | --- |
| Caller-declared affine model: constant plus weighted inputs and fixed controls | Point evaluation, analytical first sensitivities and linear-requirement audits | Supported uncertainty domains; no adjustable affine recourse or nearest-failure search |
| Heat-recovery reference model | Heat-delivery evaluation, optimized-response sensitivities, full-box audits, feasibility boundary, positive-shortfall breaking search and engineering-change comparison | Static, constant COP, two uncertain inputs and fully observed electrical-power recourse |
| Thermal-storage reference model | Fixed, causal and perfect-foresight audits; conflicting futures; equipment and earlier-information changes | Complete declared finite two-period tree; no general trajectory solver or sensitivity operator |
| Supplied startup profiles | Peak and integrated load, shared-capacity audits and comparison of fixed start schedules | Finite amplitude/duration/timing scenarios; all time segments of the declared piecewise-linear profiles; no motor dynamics or adaptive scheduler |
| Heat-cascade reference model | Minimum heating/cooling targets, maximum heat recovery, pinch locations and utility-target audits | Finite steady-state sensible-heat scenarios at fixed ΔTmin; no installed exchanger-network feasibility or sensitivity operator |

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

## Design lineage

Operange originated as `updatesupport-process` in the
[`updatesupport` repository](https://github.com/nahuaque/updatesupport/tree/08b3652bb6f8e9c7437b4329632c6e6583e9b96a/packages/updatesupport-process).
The extraction retains the MIT license and the `process_result/v1` artifact
format. It has its own release version and no compatibility import alias.


The uncertainty-set vocabulary also draws on [ROmodel, by Wiebe and Misener](https://doi.org/10.1007/s11081-021-09703-2),
which separates uncertain parameters, uncertainty sets and adjustable variables
in robust-optimization models. The implementation does not depend on ROmodel
or copy its code. Classical flexibility analysis and the related tools linked
above are foundations and complements, not capabilities claimed as new here.
