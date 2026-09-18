# Actuators, algebraic states and operating policy

A physical actuator has command permissions. An algebraic state participates in
the equations and bounds without acquiring a command rule. Both may be variables
in the same optimization problem.

`LinearProcessAdapter` declares `controls: tuple[LinearControl, ...]` and
`states: tuple[LinearState, ...]`. Its `variables` property combines them for
mathematical assembly. Contract quantities retain their `control` or `state`
roles. Bounds are identified by `control_lower:<name>`, `control_upper:<name>`,
`state_lower:<name>` and `state_upper:<name>`.

Fixed commands constrain only actuators. State equations, state bounds, equipment
limits and selected requirements still need a jointly feasible completion. An
algebraic state is not necessarily unique: feasibility reports provide a checked
completion, not a uniqueness proof or an independently commandable flow. Declare
manipulated split rates and other operating choices as actuators when their
information permissions matter. Declaring a variable as a state does not supply
missing physical equations.

## Declare network actuators explicitly

Set `ProcessNetwork(..., actuators=("fuel",))` to compile that stream as a
`LinearControl`, and every other non-input stream as a `LinearState`. Fixed
stream values become equal bounds on the corresponding variable. Use
`actuators=()` for a passive network, whose operating policy has no command rules.
Input-bound streams cannot be actuators.

Omitting `actuators` retains the historical interpretation: every non-input
stream is a control. Existing network declarations and saved identities remain
unchanged. Use an explicit tuple for new models or to migrate an existing one.

IDAES already separates actuators and states. New `export_linear()` results now
preserve those roles directly, so only original actuators need permissions.
Previously saved linear exports still load with their declared historical roles.

## Use an explicit operating specification

`OperatingSpec` packages a `RecoursePolicy` and, optionally, a saved affine
controller. It is accepted by linear and IDAES `as_claim(..., operating=spec)`.

| Specification | Meaning |
| --- | --- |
| `OperatingSpec.fixed(model.controls, commands)` | Every actuator has a prescribed value; states remain in the physical solve. |
| `OperatingSpec.dispatch(model.controls, observations)` | Each actuator may use the declared observations. Pointwise dispatch requires all true inputs. |
| `OperatingSpec.shared(model.controls)` | Commands observe no scenario inputs. They require a joint study or controller synthesis. |
| `OperatingSpec.controlled(controller, permissions=policy)` | Execute the saved rule within explicit permissions; omitted permissions come from its declaration. |

Construct `OperatingSpec(RecoursePolicy(...))` for mixed fixed/free permissions.
An explicit specification cannot be combined with `recourse` or `controller`
arguments. The existing arguments remain supported and normalize through the
same boundary. Claim binding rejects permissions on states, and tracking
objectives can target only actual actuators.

This example has one commanded fuel rate and one dependent steam rate:

```{literalinclude} ../examples/operating_roles.py
:language: python
:start-at: from operange import (
:end-before: if __name__
```

The fixed 8 MW fuel command supplies approximately 4 MW steam at every load.
Fully observed dispatch uses approximately twice the demand in fuel. The saved
controller commands fuel alone; finite replay solves and checks steam separately.

## Query scope and evidence

Evaluation, finite and continuous linear dispatch audits, objectives, relief,
operating-limit searches, and local objective sensitivities retain all state
variables and bounds. Use `DispatchChange("state_upper", ...)` or `"state_lower"`
for algebraic capacity bounds, and the existing `control_upper`/`control_lower`
variants for actuators. The independent verifier reconstructs both roles without
using the production solver or row compiler.

The [shared linear representation](linear-problem-representation.md) describes
how production queries preserve these roles through normalization, scenario
replication, relief and controller synthesis.

Saved controllers with algebraic states support **finite replay and finite
synthesis**. Every candidate fixes the actual rounded actuator commands and
checks existence of a physical state completion. A failed solve stays unresolved;
failure requires an exact bounded-linear contradiction. State-dependent controller
objectives are unsupported because a feasible completion need not be unique;
use an input/actuator objective or an explicit response model.

Continuous rounded-controller audits, synthesis and distance queries with
algebraic states remain unsupported. Generator feasibility alone does not bound
the effect of rounding throughout the domain. Existing controllers whose outputs
are expressed directly in inputs and actuators retain their continuous support.
This restriction does not affect continuous fixed-command or fully observed
linear dispatch audits, whose generator theorem covers algebraic variables too.

Existing certificate schemas retain historical keys such as `control_coordinates`
for mathematical proof coordinates. Those coordinates can now include states;
the contract's quantity roles distinguish them from actuator permissions.

## IDAES and calibration studies

`IdaesAdapter.optimize_shared_controls(domain, operating=spec)` accepts shared
permissions, including a mixture of prescribed commands and commands chosen
jointly. It records explicit permissions in the result and checks that fixed
entries survived the solver. Plain pointwise evaluation refuses an unbound shared
policy instead of optimizing each scenario independently.

Local `propagate_uncertainty(..., operating=spec)` accepts fully fixed operation
or fully observed reoptimization. Existing `fixed_controls` and `reoptimize`
arguments remain supported; mixing them with an explicit specification is an
error. IDAES does not execute affine controllers directly; supported linear
exports provide that path.

Bootstrap `as_claim`, `optimize_shared_controls` and `propagate_uncertainty`
accept the same specification. A joint bootstrap dispatch policy must explicitly
observe every required coordinate, including `parameter:<estimate-name>` latent
parameter coordinates. Permissions never expand silently when samples are added.
