# The engineering workflow

Operange asks whether a process can keep meeting its requirements as operating
conditions vary, given the controls and information available to its operators.
The result ties a verdict to a declared engineering commitment: the model,
uncertainty envelope, requirements and operating permissions.

## Start with a concrete commitment

Suppose a dryer and an evaporator share a steam header rated for 19 MW.
Their nominal demands are 10 MW and 6 MW, and each can vary by 2 MW.
Either individual peak fits; their combined 20 MW demand does not.
An audit of the full declared load envelope identifies that combination and the
shared capacity constraint. The [getting-started tutorial](index.md) implements
this study and adds a shared boiler.

The possible responses have different meanings:

| Proposed change | Engineering meaning |
| --- | --- |
| Increase header capacity | Accommodate the original load envelope with different equipment. |
| Prevent the peaks from coinciding | Restrict the allowed operating conditions; justify the revised envelope. |
| Reduce required production | Change the service commitment. |
| Change the control rule | Test whether the new rule works under the stated information permissions. |

The [engineering-change comparison](engineering-changes.md) retains the baseline
and candidate audits, records changed declarations, and distinguishes restored
service from a passing claim with revised assumptions.

## Declare the study

1. **Model and design:** balances, capacities and physical outputs, such as steam
   demand, heat delivery or stored energy.
2. **Uncertain conditions:** what can vary together, with units, bounds and the
   source of those assumptions.
3. **Operating permissions:** which controls may adjust and what information is
   available when each decision is made.
4. **Requirements:** named constraints that must remain satisfied.
5. **Query:** a point evaluation, sensitivity, audit or supported search for a
   nearby failure.

Uncertainty declarations include independent bounds (`BoxSet`), explicit finite
cases, fractions summing to one (`SimplexSet`), budgets limiting simultaneous
deviations, ellipsoids and linear coupled restrictions (`PolytopeSet`). Named
vector inputs and domain composition are available. Each model declares which
queries it supports over those domains; see the [capability table](api.md#what-the-built-in-adapters-establish).

Your model supplies the equations and consistent units. The library does not
provide thermodynamic property packages or automatic unit conversion.

## Interpret the evidence

Results contain physical quantities, constraint residuals, supported derivatives
with units, and audit verdicts with witnesses, bounds and coverage. A **witness**
is an admissible condition, or a conflicting set of futures, that demonstrates
failure under the declared operating permissions.

Execution status and engineering verdict answer different questions. A completed
calculation can establish a failed claim. A failed numerical solve alone does
not establish physical infeasibility. An unresolved audit can still retain a
verified failure, while a pass needs evidence covering its declared domain.

A finite-scenario pass covers those scenarios. A nearest failure is reported as
such only when the search bounds support it. Numerical tolerances remain part
of the result. Inspect these qualifications alongside the verdict.

## Separate adjustable operation from controller replay

An adjustable audit asks whether permitted decisions can satisfy the requirements.
A frozen-controller audit executes one specified rule. That rule may fail even
when another permitted controller would work.

The [linear dispatch guide](linear-dispatch.md) covers bounded adjustable controls.
[Frozen affine controllers](frozen-controllers.md) retain actual commands and
violations without redispatch. [Causal storage replay](storage-replay.md) also
preserves observation timing, energy state and terminal requirements over
independent two-period paths.

## Use the results downstream

The common API returns structured results and portable JSON evidence for your
own code to inspect, store and compare. Pricing, investment selection and
presentation belong to the downstream application. Repository examples can
add these layers without changing the engineering audit.

The [API guide](api.md) documents result formats and the shared, reference and
experimental namespaces. The [methods guide](methods.md) explains the connection
to classical flexibility analysis, robust optimization, sensitivity methods and
techno-economic studies, as well as the project's design lineage.
