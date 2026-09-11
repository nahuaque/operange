# Operange: process robustness and flexibility

`operange` is an independent alpha distribution for expressing
engineering claims and returning numerical evidence to downstream code. Your
model supplies the physical response; the DSL binds uncertainty, requirements
and operating permissions. Your application owns economic assumptions, selection
and reporting.

This package studies physical process robustness and operational flexibility.
Hidden-variable recomposition belongs to the main `updatesupport` package and
can be used separately when auditing aggregate economic reports in a TEA.
The [package README](https://github.com/nahuaque/operange/blob/main/README.md)
introduces the engineering workflow. The [methods guide](methods.md) compares
it with familiar methods and explains the relationship to downstream economics.

It requires Python 3.10+ and NumPy/SciPy only. From a checkout:

```bash
python -m pip install .
```

The current development version is `0.1.0`. The project is prepared for independent PyPI releases; publication is a separate step. The public entry points are documented below.

## Declare the complete engineering boundary

This example has two process users sharing a steam header and a boiler with
fixed 80% efficiency. It covers the declared utility system with closed-form
balances. It does not approximate an unspecified full plant.

```{literalinclude} ../examples/steam_header.py
:language: python
:start-at: from operange import (
:end-before: def run_example():
```

Each `Parameter` declares a name, unit, nominal value, bounds, normalization
scale and provenance. The synthetic box permits both loads to reach their upper
bounds together. All coefficients and values must use consistent units; unit
labels do not perform conversions.

## Evaluate, differentiate and audit

```python
model, loads = example()
claim = model.as_claim(loads)
evaluation = claim.evaluate_result(loads.nominal)
sensitivity = claim.sensitivity_result(
    loads.nominal,
    outputs=("steam", "fuel"),
    inputs=("dryer", "evaporator"),
    coordinate_space="physical",
)
audit = claim.audit_result()
```

Nominal demand is 16 MW of steam and 20 MW of fuel. An extra MW from either
user requires 1 MW of steam and 1.25 MW of fuel. The requested physical Jacobian
therefore has rows `(1, 1)` and `(1.25, 1.25)`. These are fixed-operation
derivatives; they do not silently reoptimize controls.

The box audit fails: peak demands of 20 MW of steam and 25 MW of fuel exceed
the declared limits of 19 MW and 24 MW. The result contains a domain member
that violates a requirement. A consumer can replay it through the same claim:

```python
witness = audit.payload.witness
replayed = claim.evaluate_result(witness.realizations[0])
checks = replayed.payload.constraint_checks
```

Always inspect `execution`, feasibility or verdict, and evidence coverage.
Successful execution can establish infeasibility or a failed claim. Conversely,
unsupported execution carries no verdict. An unresolved search or partial audit
can still retain a verified failure witness; inspect the verdict and evidence
separately from execution status.
This claim has no distance metric, so `breaking_result()` is unsupported.
Attach an explicit `NormalizedLInf` distance to search for the nearest declared
limit or positive violation over a box or polytope. The
[failure-distance guide](failure-distance.md) extends this same utility model.

## Re-audit an engineering change

For named candidate comparisons, use `EngineeringChange` and
`claim.compare_changes(...)`. The [engineering-change guide](engineering-changes.md)
compares equipment, domain and service changes, retaining every audit and marking
which passing candidates restore the original commitment.

```python
from dataclasses import replace

enlarged = replace(model, requirements=(
    AffineRequirement("steam_header", "steam", 21),
    AffineRequirement("boiler_fuel", "fuel", 26),
))
repaired = enlarged.as_claim(loads).audit_result()
```

The enlarged capacities pass over the original load envelope. A downstream
application can price that equipment change independently of this audit.

There is another way to obtain a passing claim:

```python
budgeted = model.as_claim(BudgetSet(loads, 1)).audit_result()
```

This limits the total absolute normalized deviation to one. It changes the
assumed load envelope while retaining the original capacities. It must therefore
be justified as a different operating assumption; it is not an equipment repair.

## Export the evidence

```python
serialized = audit.to_json()
restored = result_from_json(serialized)
```

The `process_result/v1` bundle carries the bound contract, named quantities and
constraints, query, numerical evidence and content-derived identities. Reading
it validates the artifact; it does not instantiate a simulator or prove the
physical model correct. Preserve the model implementation and its provenance
alongside results when independent replay is required.

For larger finite studies, use the compact transport to store the shared contract
once:

```python
compact = audit.to_json(compact=True)
restored = result_from_json(compact)
assert restored.result_id == audit.result_id
```

This uses `process_result_bundle/v1` around the same `process_result/v1` results.
The default export and existing saved bundles remain supported. Compact bundles
require the updated reader; see the [API guide](api.md) for validation details.

Run the complete example from the repository root:

```bash
uv run python -m examples.steam_header
```

The installed-wheel acceptance check copies this same consumer outside the
checkout and runs isolated Python with only the wheel and its declared
dependencies. It verifies balances, sensitivities, witness replay, re-audits,
unsupported-query behavior and compatibility with pre-refactor saved results.
CI applies this check on Python 3.10–3.13.

## Reference models and downstream code

For adjustable operation in a caller-declared model, use the
[linear dispatch guide](linear-dispatch.md). It extends the shared-utility study
with bounded boiler controls and joint equipment constraints, comparing fixed
commands with fully observed redispatch over finite scenarios.

`operange.reference` contains bounded heat-recovery, storage,
startup-profile and heat-cascade models. The repository's
`examples.heat_recovery` and
`examples.thermal_storage` modules add priced catalogs and Markdown via
`examples.reporting`. Those examples are not part of the installed
distribution. A consumer can instead use its own economic and reporting layer.

Broader simulator integration requires an adapter with explicit evaluation and
proof capabilities. Neither a general model callback nor local sensitivities
alone can certify arbitrary nonlinear whole-plant robustness.

```{toctree}
:maxdepth: 1

api
linear-dispatch
failure-distance
engineering-changes
startup
pinch
methods
```
