# Fixed affine process claims

Implemented on the experimental branch, 2026-09-11. `AffineProcessAdapter`
connects caller-declared affine responses to the uncertainty geometries and
existing evaluation, sensitivity and robustness result families. It adds no
simulator dependency, business ranking or presentation layer.

## Declare the physical response

```python
from operange import (
    AffineOutput, AffineProcessAdapter, AffineRequirement, AffineTerm,
    BoxSet, BudgetSet, Parameter,
)

loads = BoxSet((
    Parameter("dryer", "MW", 10, 8, 12, 2, "Synthetic load envelope"),
    Parameter("evaporator", "MW", 6, 4, 8, 2, "Synthetic load envelope"),
))
model = AffineProcessAdapter(
    name="Shared steam header",
    input_space=loads.space,
    outputs=(AffineOutput(
        name="steam_demand", unit="MW", physical_kind="thermal_power",
        offset=0,
        terms=(AffineTerm("dryer", 1, "MW/MW"),
               AffineTerm("evaporator", 1, "MW/MW")),
        provenance="Sum of declared process loads",
    ),),
    requirements=(AffineRequirement("header_capacity", "steam_demand", 19),),
)
claim = model.as_claim(loads)
assert claim.audit_result().payload.verdict == "fail"
assert model.as_claim(BudgetSet(loads, 1)).audit_result().payload.verdict == "pass"
```

Each output has the form `offset + sum(coefficient * variable)`. Variables refer
to named physical inputs or declared scalar controls. The offset uses the output
unit; coefficient units must explicitly match `output_unit/input_or_control_unit`,
including ratios such as `MW/MW`. No automatic unit conversion, recursive output
references, expression parsing, nonlinear terms or property calculations occur.
Terms, offsets, provenance, units and coordinate metadata enter the result's
bundled process contract.

`AffineRequirement` compares an output with a limit in that output's unit.
`relation="le"` declares a maximum; `"ge"` declares a minimum. Both become a
signed residual that must be at most `tolerance`, which defaults to `1e-8` in
the output unit. Use two requirements for an interval. The positive
`residual_scale` defaults to one output unit and controls normalization for the
support calculation; it does not alter the physical requirement.

`as_claim(domain, requirements=(...))` can select a subset of declared
requirements. The equations remain the same. Point evaluation checks only the
selected service requirements, so an unselected requirement cannot change an
audit verdict.

## Fixed operation

Models without controls use `RecoursePolicy("fixed", ())`, meaning there are
no operating adjustments. Empty policies are allowed only in fixed mode.

For controls, supply scalar `QuantitySpec` declarations with `role="control"`
and include them in affine terms. Bind a `RecoursePolicy("fixed", ...)` with
one `DecisionRule` per control, each specifying a `fixed_value` and no observed
fields. Every declared control must be fixed. Missing values, partial control
coverage, static adjustable, causal and perfect-foresight operation return
unsupported queries. Fixed values affect the response and its content identity.

A failure is a `fixed_policy_failure` witness: the unique response permitted by
this operating contract violates a selected requirement. It does not establish
that a different control setting, controller or design would also fail.

## Evaluation and sensitivity

```python
point = {"dryer": 10, "evaporator": 6}
evaluation = claim.evaluate_result(point)
derivative = claim.sensitivity_result(point, coordinate_space="normalized")
assert evaluation.payload.feasibility == "feasible"
assert derivative.payload.derivative.values == ((2.0, 2.0),)
```

Evaluation first verifies domain membership, then calculates the affine
outputs with the fixed controls. Outside points are invalid; unknown membership
or unrepresentable arithmetic returns unresolved execution without an operating
solution. There is no optimizer in point evaluation.

`feasibility="feasible"` means the declared affine response equations are
satisfied. Service requirements are separate `constraint_checks` and can be
violated at that same point. This scope is recorded in the contract; no
unmodeled equipment feasibility or physical law is inferred from an affine
formula. Sensitivities remain meaningful at a defined response that violates
service.

Sensitivity supports analytical Jacobians and directional derivatives, selected
named `inputs` and `outputs`, and physical or normalized coordinates. Controls
are held fixed. Physical derivatives are the declared coefficients; normalized
derivatives multiply each input column by that domain coordinate's scale.
Changing the domain's nominal/scale metadata never changes the physical model.
Directional queries take a finite `direction` tuple in the selected coordinate
order and return output change per dimensionless path parameter.

The derivative contract explicitly uses `response="fixed"` and
`perturbation_scope="ambient"`: it differentiates the declared affine map in
its surrounding coordinate space. For a simplex, for example, the independent
partial derivatives exist for that map even though changing only one fraction
would leave the simplex. No tangent feasibility or feasible perturbation is
asserted. Domain-restricted derivatives, optimized responses and higher-order
operators are unsupported. The base realization must still be a verified
domain member. The affine derivative is the same on either side; `side` retains
the usual API choices without implying admissible directions.

## Full-domain audit

The adapter supports audits wherever a domain advertises normalized
`maximize_linear` and provides complete nominals and scales:

| Domain | Audit support |
| --- | --- |
| `BoxSet` | Separable endpoint extrema |
| `FiniteSet` | Complete enumeration, with explicitly declared normalization |
| `SimplexSet` | Analytical vertex support |
| `BudgetSet` | Analytical budget allocation |
| `EllipsoidSet` | Analytical ellipsoidal support |
| `PolytopeSet` | Checked primal/dual bounds; finite-envelope bounds survive solver failure |
| `Intersection`, `Union`, `Product` | Membership/evaluation only; audit unsupported |

Box and finite support use the existing dimensionless `LinearSupport` record.
Finite domains without normalization retain membership and enumeration, physical
point evaluation and physical sensitivities. They do not invent a baseline or
scale to enable an audit or normalized derivative.

For requirement residual `r(u)`, physical input coefficients `a`, domain
nominal `n`, input scales `s`, and requirement residual scale `q`:

```text
z_i = (u_i - n_i) / s_i
w_i = sign * a_i * s_i / q
r(u) = r(n) + q * sum(w_i * z_i)
```

The fixed control contribution is retained in `r(n)`. A support bound therefore
gives a physical residual upper bound. The adapter adds `q * support.tolerance`
as a numerical guard and records it separately from the requirement tolerance.
A very large residual scale can make this guard too coarse for a tight physical
requirement; the result then remains inconclusive. Overflow or coefficient
underflow also remains unresolved.

The decision rules are deliberately separate:

- **Pass:** every selected requirement has a verified residual upper bound at
  or below its tolerance, with a verified supporting response.
- **Fail:** a candidate passes domain membership and direct physical evaluation
  establishes a selected requirement violation under the fixed operation.
- **Inconclusive:** neither obligation is established. A positive upper bound
  alone is not a failing realization, and an open support gap need not imply
  failure.

An optimal support solution is not required when looser bounds already establish
a pass. Conversely, a verified violating realization suffices for failure even
when other support calculations are unresolved. The audit checks returned domain
identity, objective coefficients, candidate membership and attained scores.
Upper-bound evidence remains the responsibility of the domain implementation.
These are numerical certificates for declared affine equations, not exact
arithmetic proofs or validation of a physical model against observations.

Evidence with `subject="requirement_bound"` identifies its requirement in
`details.requirement`. It records `attained_residual` and `residual_upper` as
unit-bearing measurements, the numerical guard, and the complete normalized
support record. Supporting point evaluations and any failure witness are bundled
in the ordinary `RobustnessResult`. No fourth process result family is added.

The [failure-distance extension](../docs/failure-distance.md) supports boundary
and positive-violation searches over `BoxSet` and `PolytopeSet` when a
`NormalizedLInf` distance is explicitly attached. It checks physical candidates,
exact domain rows, and lower/infeasibility certificates independently of solver
termination. Other uncertainty geometries remain unsupported for these searches.

## Worked comparison and portability

The [utility example](../examples/affine_utility.py) uses a labeled
vector of dryer and evaporator loads with the same physical model for every
domain. The nominal total is 16 MW and header capacity is 19 MW.

| Domain assumption | Maximum demand | Verdict |
| --- | --- | --- |
| Full box | 20 MW | Fail |
| Deviation budget 1 | 18 MW | Pass |
| Coupled normalized excursions summing to at most 1 | 18 MW | Pass |
| Unit-radius spherical normalized ellipsoid | `16 + 2 sqrt(2)` MW | Pass |
| Listed nominal/single-process-peak scenarios | 18 MW | Pass |

These results compare declared assumptions, not empirical confidence levels.
Run the example from the repository root to emit portable evaluation,
sensitivity and audit bundles for each domain:

```bash
uv run python examples/affine_utility.py
```

`result_from_json` validates the existing `process_result/v1` bundles without
executing a model or solver. Declarations and fixed controls affect contract
digests; outputs and evidence affect result digests. This slice does not add
automatic executable adapter reconstruction or a frozen-controller replay API.
