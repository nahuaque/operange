# Existing IDAES and Pyomo flowsheets

Use `operange.integrations.idaes` to attach operating studies to an existing
IDAES/Pyomo model. The supplied flowsheet remains the physical model: its unit
equations, property calculations, connections and operational constraints are
not reimplemented in Operange. The integration is optional:

```bash
python -m pip install 'operange[idaes]'
```

This installs IDAES, Pyomo and the HiGHS Python solver. HiGHS runs the affine
example; nonlinear models need a separately installed compatible solver, such
as Ipopt, selected with `solver="ipopt"`. Importing `operange` or the binding
declarations does not import either modelling package. Base installations can
load the resulting JSON reports without IDAES, Pyomo or a solver.

## Bind an existing model

The complete [worked example](examples.md) is
`examples/idaes_flowsheet.py`. It builds two fixed-efficiency boiler units using
IDAES `SkeletonUnitModel` blocks and a `FlowsheetBlock`. These simple unit laws
make the affine comparison explicit; the integration also accepts nonlinear
algebraic models and existing IDAES unit/property packages for numerical queries.

```python
from operange import Coordinate, ParameterSpace, FiniteSet, Scenario
from operange.integrations.idaes import (
    IdaesAdapter, IdaesVariable, IdaesRequirement,
)

# m is the caller's constructed, consistently unit-labelled Pyomo ConcreteModel.
# Initialize a nonlinear flowsheet before binding it if its solver needs that.
adapter = IdaesAdapter(
    model=m,
    name="Two-boiler plant",
    input_space=ParameterSpace((
        Coordinate("load", "MW", "Required steam duty", nominal=16, scale=4),
    )),
    inputs=(IdaesVariable("load", "fs.load", "MW"),),
    controls=(
        IdaesVariable("fuel_a", "fs.boiler[a].fuel", "MW"),
        IdaesVariable("fuel_b", "fs.boiler[b].fuel", "MW"),
    ),
    outputs=(
        IdaesVariable("steam", "fs.steam", "MW"),
        IdaesVariable("fuel", "fs.fuel", "MW"),
    ),
    requirements=(IdaesRequirement("service", "fs.service", "MW", 0),),
    tolerance=1e-8,
    solver="highs",
    provenance={"study": "Utility capacity review"},
)
cases = FiniteSet(adapter.input_space, (
    Scenario("normal", {"load": 16}, "Normal operation"),
    Scenario("peak", {"load": 20}, "Peak operation"),
))
claim = adapter.as_claim(cases)
point = claim.evaluate_result({"load": 16})
report = claim.audit_result()
portable = report.to_json(compact=True)
```

Bindings use scalar Pyomo `ComponentUID` paths, including indexed component
entries. Units must match the component's native Pyomo unit spelling (`"1"`
for dimensionless quantities). Make conversions explicit in the source model.
The constructor checks the source model's unit consistency.

Inputs bind Vars or mutable Params with domain `Reals` or `Any` and no custom
validation callback. Declare physical input limits as constraints or Var bounds.
Controls bind the actual manipulated
Vars. Other unfixed Vars remain algebraic states and appear as `state:<CUID>` in
results. Their uniqueness is not assumed or established. A source actuator may
be fixed at its nominal value: binding it as a control makes the claim's explicit
operating permissions authoritative. Other fixed source variables stay fixed.

Requirements identify active, one-sided service constraints. Equality laws and
ranged operating constraints remain physical constraints. Selecting fewer
requirements deactivates only the omitted, explicitly labelled service rows;
all other active constraints and all variable bounds still apply. Choose these
labels deliberately: the integration cannot infer which source inequalities
represent service promises rather than physical limits.

The adapter snapshots the model at construction and solves a fresh clone per
evaluation. Subsequent caller edits do not change an existing claim. Construct
a new adapter after calibration, equipment changes or initialization changes.
The source manifest records active equations, variable bounds/fixed values and
initial values, parameter values, named expressions, objectives, backend versions,
binding roles, solver options and caller provenance. These identify the recorded
study; they are not an executable serialization of an arbitrary IDAES model.

## Numerical evidence and coverage

Expand network Arcs into algebraic constraints before binding (Pyomo's
`network.expand_arcs` transformation). Ports and component references are
supported. Active unexpanded Arcs are rejected so connections cannot be omitted
from the physical checks. Compiled AMPL external property functions can be used
numerically when their libraries are available; record the library revision in
provenance. Their names and library paths are recorded, not their executable
contents. Python external callbacks are unsupported in this first binding.

The first integration supports continuous algebraic variables with fixed or
fully observed single-stage actuator permissions. Partial-observation and causal
policies, integer/disjunctive constraints, dynamic discretization, and automatic
initialization are outside this binding's scope.

A solver must return a fresh solution with an accepted termination condition.
Operange substitutes its variable values into a separate copy of the original
model, checking every included constraint, variable bound and fixed command.
Constraint tolerances are absolute in each row's native unit. The adapter-wide
tolerance applies to non-service rows; each requirement declares its own.
Variable bounds and fixed values have zero tolerance. The source optimization
retains its own equations and objective; tolerances describe acceptance checks,
not changes to the solver's equations.

| Outcome | Meaning |
|---|---|
| Evaluation `feasible` | A numerical candidate satisfies all included checks at this point. |
| Finite audit `pass` | Every declared scenario has such a candidate; coverage is limited to those scenarios. |
| Evaluation `unknown`, audit `inconclusive` | Missing solver, failed solve, bad residual, undefined value, or another unresolved scenario. |
| Nonlinear infeasibility / global optimality | Never asserted by this adapter, including when a solver reports infeasibility or optimality. |

Point evaluation can use any domain with supported membership checks. Whole-domain
audits require the built-in `FiniteSet`. The native adapter does not supply
continuous-envelope proofs, sensitivity/distance queries, saved executable
controllers or independent nonlinear certificates. `verify_result()` returns
`unsupported` for these numerical IDAES reports. Loading and integrity-checking
their JSON is still supported and does not execute the model.

## Checked affine export

```python
from operange import verify_result
from operange.integrations.idaes import IdaesLinearExport

exported = adapter.export_linear()
affine_claim = exported.as_claim(cases)
certificate = affine_claim.audit_result()
verification = verify_result(certificate)
restored = IdaesLinearExport.from_json(exported.to_json())
```

The exporter walks supported source arithmetic with exact rational coefficients.
It accepts sums, negation, multiplication by constants and division by nonzero
constants, and rejects nonlinear expressions, unsupported functions and any
coefficient arithmetic that cannot be represented exactly by the target floats.
It exports the expression tree Pyomo has already constructed; arithmetic rounded
by Python/Pyomo before binding is part of that supplied model. It never replaces
nonlinear equations by a local linearization or fitted surrogate.

Input Vars and mutable input Params remain symbolic, even if fixed or assigned
nominal values in the source. Products involving uncertain coefficients and free
variables are rejected. Every free actuator and algebraic state needs finite,
constant bounds. Fixed source states become constants; their bounds remain
checked. The active objective, if present, must also be affine. Source equation
tolerances and requirement selection are retained in the exported model.

The result contains an ordinary `LinearProcessAdapter`, the source contract and
an actuator map, under `idaes_linear_export/v1`. Its `as_claim(..., recourse=...)`
accepts permissions for the original actuators, adding algebraic-state recourse.
Exported state variables are represented as linear controls for solving: a
controller synthesized directly on that adapter still includes algebraic rates
and is not automatically an executable physical controller.

The existing exact verifier checks certificates **on the exported coefficient
model**. It does not independently verify the Pyomo-to-coefficient translation,
the validity of the source physics, or any approximation error in a source model.
Source fingerprints in output provenance identify the translation being used.
Supported linear models can use the existing continuous audits, diagnosis,
relief and operating-limit queries with their documented evidence scope.

## Reuse upstream reconciliation and estimation

```bash
uv run --extra idaes python examples/idaes_flowsheet.py
# Requires the Ipopt executable on PATH:
uv run --extra idaes python examples/idaes_flowsheet.py --reconcile
```

The second command calls Pyomo's public `Experiment` / `Estimator.theta_est`
interfaces directly. It reconciles noisy fuel/steam measurements with a known
boiler efficiency, then estimates efficiency from a separate calibration
campaign. It builds a fresh flowsheet with the estimate and repeats the study.
Operange introduces no replacement estimator or data-management framework.

The initial 80% / 50% efficient plant uses 23 MW fuel at 16 MW demand and 31 MW
at 20 MW demand. Both native numerical and exported affine studies pass. The
calibration gives approximately 74% efficiency for boiler A: the 20 MW demand
cannot be met under the 32 MW shared fuel limit. The native result is inconclusive;
the affine path proves failure with an independently verified certificate.

Measurement weights, fit results and calibration provenance are preserved.
Statistical estimation uncertainty is not automatically converted into a hard
operating envelope, and fitted coefficients do not imply certified physical
accuracy. Supply a justified uncertainty set separately.

Use the current [IDAES reconciliation guidance](https://idaes-pse.readthedocs.io/en/stable/how_to_guides/workflow/data_rec_parmest.html)
and [Pyomo parmest interfaces](https://pyomo.readthedocs.io/en/stable/explanation/analysis/parmest/index.html).
Older examples using IDAES DMF are not the integration target; DMF is no longer
supported. Future nonlinear robust optimization should assess
[PyROS](https://pyomo.readthedocs.io/en/stable/explanation/solvers/pyros/index.html)
against the required operating permissions and evidence before adding a backend.
