# Choose shared commands under equipment uncertainty

The [joint calibration workflow](joint-calibration-uncertainty.md) can choose
one actuator vector that satisfies every declared parameter-draw/operating-case
combination. The optimizer uses the active objective in the original Pyomo
model. An existing fuel, utility-cost or production objective therefore retains
its native expression, unit and minimization/maximization sense.

```python
plan = uncertainty.optimize_shared_controls(
    adapter,
    operating_cases,
    parameter_map={
        "fs.boiler.efficiency": "fs.boiler.efficiency",
        "fs.boiler.heat_loss": "fs.boiler.heat_loss",
    },
)
if plan.execution != "completed":
    raise RuntimeError(plan.diagnostics)

print(dict(plan.commands))
print(plan.objective_value, plan.objective["unit"])
print(plan.audit.payload.verdict)
```

`uncertainty` must be a completed `CalibrationBootstrap`, and `operating_cases`
must be a `FiniteSet` in the adapter's original input space. The same explicit
parameter-map, native-unit and per-draw bound validation applies as for
`uncertainty.as_claim()`. Whole vectors, duplicate draw labels and operating-case
labels are preserved. The original adapter and calibration remain unchanged.

For an existing finite input domain without calibration, call the adapter
directly:

```python
plan = adapter.optimize_shared_controls(finite_domain)
```

## Operating meaning

All declared actuators receive one common command, chosen before observing
either the operating case or the equipment parameters. Algebraic states can
differ between scenarios according to the source equations. Every physical
actuator must be declared as a control; the adapter does not infer actuator roles
or establish uniqueness of the remaining states.

Every active source requirement, equation and variable bound applies in every
scenario. The source must have at least one actuator and exactly one active
objective. Source-fixed actuator values are starting guesses; the optimization
releases declared actuators to choose their common values. Other fixed variables
remain fixed unless explicitly bound as scenario inputs.

Aggregation is explicitly **worst case**:

- A minimization objective minimizes its largest scenario value.
- A maximization objective maximizes its smallest scenario value.

There are no inferred scenario probabilities, expected costs or confidence
levels. Duplicate scenarios retain their identities without weighting the
objective differently. A policy that adjusts to measured demand is outside this
first version; every declared command is shared across all inputs.

## Computation and evidence

Operange builds a Pyomo model containing one source clone per scenario, shared
actuator variables, command-equality constraints and a worst-case objective.
The adapter's solver, options and explicit IDAES initialization steps are reused.
Initialization remains confined to disposable clones and only supplies starting
values. The model grows with the number of scenarios; no sampling reduction or
survivor filtering is performed.

After a fresh solver candidate is returned, command-equality errors are checked
against the adapter tolerance. The exact same command numbers are then
substituted into every scenario. The existing evaluation checker copies the
candidate states into separate, unmodified source-model clones and checks all
equations, active requirements, fixed values and variable bounds. This is
numerical substitution; it does not run a second optimization for each scenario.
Each objective value is recomputed from its original source expression. The
aggregate value is taken from these checked values, rather than from the
solver's auxiliary worst-case variable.

`execution == "completed"` means every declared scenario has a checked feasible
candidate at the shared commands. `objective_value` is the attained worst-case
value. Neither completion nor a solver's `optimal` status certifies a global
nonlinear optimum. Failed solves, initialization errors, missing candidates and
source-check violations remain unresolved, with diagnostics and any available
partial audit retained. An unresolved result can contain candidate `commands`;
check execution before using them as an accepted plan.

To independently **re-solve** the operating scenarios at the selected commands,
use the existing fixed-command path:

```python
replay = uncertainty.as_claim(
    adapter,
    operating_cases,
    parameter_map=parameter_map,
    fixed_controls=plan.commands,
).audit_result()
```

The finite numerical evidence does not establish continuous uncertainty
coverage, plant reliability, nonlinear infeasibility or an independently verified
mathematical certificate.

## Portable result

```python
from operange.integrations.idaes import IdaesSharedControlResult

saved = plan.to_json()
restored = IdaesSharedControlResult.from_json(saved)
```

`idaes_shared_controls/v1` records the original contract, finite domain, objective,
commands, per-scenario objective values, attained aggregate, solver termination,
initialization/diagnostic evidence and fixed-command audit. Calibration
provenance links the uncertainty result, parameter map and draw/case identities.
The audit uses the existing compact process-result transport.

Loading requires only Operange's base dependencies. The reader validates content
identity and consistency between commands, scenarios, objective values and the
audit. It does not execute or independently verify the source equations.

## Boiler comparison

```bash
uv run --extra idaes python examples/idaes_uncertainty.py \
  --optimize-shared --samples 12 --seed 7 --shared-output shared-plan.json
```

The synthetic study first chooses a command using the fitted nominal parameters
and tests it against the joint draws. It then chooses a shared command across all
12 draws and both demand cases. With the supplied data, nominal fuel is about
13.4996 MW and its joint audit is inconclusive. The shared command is about
13.5548 MW, with all 24 labelled scenarios passing numerical checks.
