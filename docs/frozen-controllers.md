# Frozen controllers

A frozen controller fixes the operating rule and its coefficients. Its commands
can still respond to the observations that the claim permits. Replaying it
executes those commands directly and checks the physical response, retaining
equipment and service violations without redispatching or clipping the commands.

The linear implementation binds caller-supplied affine rules to
`LinearProcessAdapter`. It supports point evaluation at verified domain members
and complete finite-scenario audits. It does not fit controllers, maintain dynamic
state, synthesize causal policies, or establish continuous-domain robustness.
The [causal storage replay guide](storage-replay.md) adds a separate bounded
two-stage controller with explicit observation signals and carried energy state.

## Two boilers and a fuel limit

Consider two boilers with steam capacities of 12 MW and 10 MW, efficiencies of
80% and 50%, and a shared 32 MW fuel supply. The combined demand reaches 20 MW.
These are synthetic engineering values.

| Operating rule at the combined peak | Boiler A / B | Fuel | Outcome |
| --- | --- | --- | --- |
| 50/50 allocation | 10 / 10 MW | 32.5 MW | Shared fuel limit violated |
| 60/40 allocation | 12 / 8 MW | 31 MW | All included constraints satisfied |

The adjustable audit also passes over the four declared scenarios. A failure
of the 50/50 controller therefore does not establish physical infeasibility
under other operating rules.

The [complete consumer example](https://github.com/nahuaque/operange/blob/main/examples/frozen_controllers.py)
declares the physical model and compares both controllers with the adjustable
benchmark. It runs using only the installed API:

```bash
uv run python -m examples.frozen_controllers
```

## Declare and bind a rule

```python
from operange import AffineControlRule, AffineController, AffineTerm

controller = AffineController(
    "60/40 allocation",
    tuple(
        AffineControlRule(
            control=name,
            unit="MW",
            offset=0,
            terms=(
                AffineTerm("dryer", share, "MW/MW"),
                AffineTerm("evaporator", share, "MW/MW"),
            ),
        )
        for name, share in (("boiler_a", 0.6), ("boiler_b", 0.4))
    ),
    provenance="Synthetic caller-supplied allocation; no fitting",
)
claim = model.as_claim(domain, controller=controller)
audit = claim.audit_result()
```

`AffineControlRule(control, unit, offset, terms=(), stage="operation")` computes
`offset + sum(coefficient * observation)` in physical units. An empty term tuple
specifies a constant command. Terms reuse `AffineTerm`; coefficient units must
match the control unit divided by the input unit. Inputs must be declared
process observations. Other controls, hidden observations, and future stages
cannot be used as inputs.

`AffineController(name, rules, provenance)` requires one rule per model control.
The default binding permits exactly the observations named by each rule. Pass
`recourse=` explicitly to bind against an existing permission declaration, or
use `existing_claim.with_controller(controller)` to preserve that claim's domain,
requirements and permissions. A controller must respect fixed commands and may
use only the observations allowed for each control. Partial observation is
supported for these explicit rules; the adjustable linear solver retains its
fully observed scope. Only the single `operation` stage is supported here.

`controller.commands(observations)` evaluates the rule without a physical model.
It reads only its named observations and returns physical float commands. Model
binding validates units and permissions; model evaluation additionally requires
all physical inputs for domain membership and response calculations.

## Freeze, load and replay

```python
from operange import FrozenController

frozen = claim.freeze()
loaded = FrozenController.from_json(frozen.to_json())
assert loaded.frozen_id == frozen.frozen_id

response = loaded.evaluate_result({"dryer": 12, "evaporator": 8})
replayed_audit = loaded.audit_result()

# A different scenario set is a new audited domain, with the same controller.
held_out_audit = loaded.as_claim(held_out_domain).audit_result()
```

The `frozen_controller/v1` artifact contains the complete physical model,
controller definition, domain, operating permissions, selected requirements,
units, numerical settings and any attached `NormalizedLInf` distance declaration.
Distance declarations are preserved but controller distance searches remain
unsupported. The nested `affine_controller/v1` declaration also has its own
identity and standalone `to_json` / `from_json` methods.

Loading checks identities and reconstructs the supported declarative binding,
including a comparison of the reconstructed contract with the saved contract.
It runs neither a controller nor an optimizer and accepts no arbitrary callbacks.
Content identities do not authenticate authorship or establish physical truth.
Keep the compatible model implementation alongside artifacts for independent
replay. Consumer-defined adapters and domain implementations cannot be frozen
through this format.

Freezing requires an explicit controller, but it does not require a passing
audit: failed controllers are useful reproducible counterexamples. The legacy
`reference.FrozenHeatRecoveryContract` remains a separate static-recourse format.

An unseen point outside the saved domain is rejected by `evaluate_result`.
`as_claim(new_domain)` explicitly binds the same model, rule and permissions to
new input support. A new finite audit covers only those declared scenarios;
it does not inherit coverage from the original audit. The original frozen
artifact remains unchanged.

## Evidence and comparisons

Controller evaluation uses the existing `EvaluationResult` and audits use
`RobustnessResult`. Controller manifests live in the contract's `operating`
declaration, so `compare_changes` identifies controller revisions as operating
changes while retaining the same physical model and service commitment.

Every executed response records the controller identity, observations used by
each rule, actual commands, physical outputs and constraint residuals. A finite
audit can carry a `fixed_policy_failure` witness for any violated included
constraint, including an equipment limit. An infeasible result concerns the
uniquely prescribed commands; it makes no claim about other controllers. Compare
an independently adjustable claim separately when that benchmark is useful.

Rules evaluate affine sums using exact rational values of the declared floats,
then round each command once to a physical float. Physical checks use those
actual commands, with the existing directed residual arithmetic and declared
requirement tolerances. Control bounds have zero tolerance. There is no implicit
unit conversion, saturation or replacement dispatch. Overflow or an unexportable
response remains unresolved; an unresolved case does not erase a verified
failure at another case. Standard and compact result exports preserve these
semantics and the existing `process_result/v1` identities.

Causal storage controllers use the same frozen envelope and result families.
See [storage replay](storage-replay.md) for signal rules, initial and terminal
energy, ordered paths and explicit tree rebinding.
