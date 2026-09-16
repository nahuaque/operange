# Guaranteed operating limits

`claim.operating_limit("production")` finds how far a declared production or
load range can extend while meeting the claim throughout its other uncertainty.
It retains a checked achievable bound, an excluded endpoint when a failure
proves one, and the corresponding physical audits and limiting realization.

## Declare the operating range

Choose a top-level scalar `Parameter` in a `BoxSet`. Its lower bound is the
operating floor and its upper bound is the search ceiling. The parameter can
represent throughput in physical units or a dimensionless load multiplier.
The model must state how it affects physical loads, such as
`steam_demand = 2 * production + auxiliary_demand`.

```python
from operange import OperatingLimitResult, verify_result

result = claim.operating_limit(
    "production",
    tolerance=1e-6,       # absolute tolerance in the production parameter's unit
    max_iterations=64,   # midpoint audits, in addition to the two endpoint audits
    max_vertices=256,    # complete generator budget for adjustable dispatch
    backend="scipy",
)
if result.lower is not None:
    guaranteed = result.as_claim()
    audit = guaranteed.audit_result()
    print(result.lower, result.unit)
```

A candidate endpoint `r` means **every production rate between the original
floor and `r`**, jointly with every allowed value of the other inputs. Each
trial changes only that coordinate's upper bound. If needed, its domain nominal
value is clipped to the smaller endpoint so the domain declaration remains
valid. Physical equations, tolerances, equipment, other uncertainty, permissions
and controller coefficients stay the same. The original claim is immutable.

These nested ranges make the feasibility question monotone without assuming
that individual operating points are monotone. For example, an exact fixed
dispatch might meet demand at 10 MW but fail at 5 MW through excess production.
A query with a 5 MW floor must report that failure; it cannot promise the entire
5–10 MW range merely because its upper endpoint works.

This initial implementation supports `LinearProcessAdapter` with fixed commands
or fully observed static dispatch, and saved affine controllers, including
[uncertain measurements](measurement-uncertainty.md). The selected coordinate
cannot be a declared sensor-error input. A multiplier must enter the existing
affine equations explicitly; products of two varying inputs are outside this
model class. Other domain geometries and two-dimensional operating maps remain
future work.

## Compare saved and adjustable operation

The example has two boilers with steam capacities of 12 and 10 MW, efficiencies
of 80% and 50%, and a shared 32 MW fuel supply. Production requires 2 MW of
steam per t/h; independent auxiliary demand ranges from 1 to 2 MW. Search
covers production from 0 to 12 t/h.

```{literalinclude} ../examples/operating_limits.py
:language: python
:start-at: def example():
:end-before: def run_example():
```

```python
adjustable, controller = example()
saved_limit = adjustable.with_controller(controller).operating_limit("production")
adjustable_limit = adjustable.operating_limit("production")
```

| Operating policy | Guaranteed range endpoint, approximately | Limiting condition |
| --- | --- | --- |
| Saved 50/50 boiler allocation | 8.846154 t/h | At 2 MW auxiliary demand, the shared fuel supply is exhausted. |
| Adjustable dispatch | 9.25 t/h | At 2 MW auxiliary demand, boiler A supplies 12 MW and boiler B 8.5 MW, consuming 32 MW of fuel. |

The bounds include the declared service and equipment tolerances. The saved
controller's failure concerns only its prescribed commands. The adjustable
upper bound needs a checked infeasibility certificate, establishing that no
permitted dispatch can handle the limiting realization. Adjustable coverage
between generators establishes existence of real-valued commands; it does not
supply an executable controller. The query never synthesizes or retunes a rule.

For adjustable dispatch, `limiting_constraints` identifies the constraints in
the joint infeasibility witness. It is not necessarily a minimal conflict or
a unique bottleneck. For a saved controller, these are the requirements or
equipment bounds violated by its actual commands. `limiting_witness` retains
the true physical realization, including sensor errors where declared.

```bash
uv run python -m examples.operating_limits
```

## Read the bounds and termination status

`lower` and `upper` are floats in `unit`, or `None` when no corresponding proof
exists. A lower bound certifies the whole operating range up to that endpoint.
An upper bound is an endpoint whose range has a checked failure. It is not a
promise that all individual rates above that number fail.

| `resolution` | Meaning |
| --- | --- |
| `limit_bracketed` | Passing and failing endpoints differ by at most the requested absolute tolerance. |
| `ceiling_verified` | The entire caller-declared search range passes. `lower` is the ceiling and `upper` is `None`; no global maximum is asserted. |
| `baseline_infeasible` | The floor already fails for an allowed realization. No range beginning there is guaranteed; `lower` is `None`. Higher isolated operating points may still work. |
| `unresolved` | Numerical uncertainty, an iteration limit, or floating-point resolution prevented a conclusive search. Any established bounds remain available. |
| `not_assessed` | Invalid or unsupported query. Read `execution` and `diagnostics`. |

`execution="completed"` includes both a successful bracket and a certified
failure at the floor. It does not by itself mean that an operating range is
achievable. Check `lower` for that promise.

The search audits the floor and ceiling, then bisects a passing/failing interval.
An inconclusive audit stops refinement and cannot become an upper bound.
`max_iterations=0` performs only the endpoint checks. An adjustable box that
exceeds `max_vertices` is rejected before any audit; there is no sampling
fallback. Saved-controller audits use their existing whole-domain support
bounds and do not enumerate corners. CVXPY is optional for adjustable dispatch;
saved-controller searches and independent verification need no optimizer.

## Export, verify and replay

```python
document = result.to_json(compact=True)
loaded = OperatingLimitResult.from_json(document)
verification = verify_result(document, expected_contract_id=claim.contract.ref.artifact_id)
```

The `operating_limit/v1` report contains the original contract, query, derived
bounds and policy scope, every trial endpoint and audit, and diagnostics.
Compact export uses compact transports for the embedded audits. Loading checks
structure and identities; it does not execute the model or verify proof math.

`verify_result` independently checks that every trial is the permitted nested
range of the original contract. Passing trials require complete physical audit
certificates. Excluded trials require an independently checked infeasible
realization; complete coverage is unnecessary for that side. Unresolved trials
assert no bound. The report's established bounds can therefore verify even when
the search stopped early. Verification does not certify that numerical search
logs describe a particular solver's execution.

The same report works with `python -m operange.verify result.json`. Verification
calls neither the search nor a production audit or optimizer.

`loaded.as_claim()` reconstructs the largest checked achievable range. For a
saved controller, that claim can be frozen and replayed normally. It retains
the original coefficients, measurement maps and observation permissions.
Distance metrics are separate from this query; attach one explicitly for
distance searches on the reconstructed claim.
