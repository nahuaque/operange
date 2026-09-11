# Causal storage replay

Freeze an explicit controller for the two-period thermal-storage reference model,
then replay its preparation and event commands with carried energy state. The
controller uses only its declared preparation signal before the event. Demand
and recovered heat become available at the event stage. Replay never solves a
new dispatch or clips commands to make them feasible.

## Declare, freeze and replay

```python
from operange import FrozenController
from operange.reference import (
    ScenarioTree, StorageController, StorageDesign, StoragePreparationRule,
    StorageScenario, ThermalStorageClaim,
)

model = ThermalStorageClaim(
    "Demand and mandatory heat recovery",
    StorageDesign(capacity_mwh=2, discharging_power_mw=2),
    ScenarioTree((
        StorageScenario("Demand", 1, 0, "Synthetic demand event"),
        StorageScenario("Recovery", 0, 1, "Synthetic recovery event"),
    )),
)
controller = StorageController(
    "Prepare at 1 MW",
    (StoragePreparationRule("unrevealed", 1),),
    provenance="Synthetic prescribed rule; no fitting",
)
frozen = model.as_claim(controller=controller).freeze()
loaded = FrozenController.from_json(frozen.to_json())
audit = loaded.audit_result()  # pass over the original two paths

held_out = ScenarioTree((
    StorageScenario("Later demand", 1.25, 0, "Synthetic held-out path"),
))
replayed = loaded.as_claim(tree=held_out).audit_result()  # fixed_policy_failure
```

Each default period lasts one hour. The initial store is empty. Preparation at
1 MW stores 1 MWh. The original demand path finishes at 0 MWh; the original
recovery path finishes at 2 MWh. The later 1.25 MW demand requires 1.25 MWh, so the
saved rule fails its energy checks even though installed discharge power is
sufficient. The response retains the prescribed discharge and the calculated
negative terminal energy as evidence of failure.

The [complete consumer example](https://github.com/nahuaque/operange/blob/main/examples/storage_replay.py)
also compares 0.5 MW and 1 MW preparation with an independently optimized causal
benchmark, and demonstrates an unseen preparation signal:

```bash
uv run python -m examples.storage_replay
```

## Controller and information contract

`StoragePreparationRule(observation, charge_mw)` maps one nonempty preparation
signal to a prescribed charging command in MW. `StorageController(name,
preparation_rules, provenance, default_preparation_charge_mw=None)` contains
unique signal rules. A default is optional and explicit; an empty rule table
requires a default. Negative or excessive commands remain representable so their
physical failures can be audited.

Preparation receives only `preparation_observation`. Paths with the same signal
therefore use identical commands, regardless of event name, demand, recovered
heat, or position in the tree. Signal labels and their availability are part of
the declared model input; replay does not establish their accuracy in practice.

The saved `event_rule="net_heat_balance"` computes:

```text
event_charge    = max(recovered_heat - demand, 0)
event_discharge = max(demand - recovered_heat, 0)
```

This is the event action determined by exact heat service, mandatory heat
recovery, no dumping, and no simultaneous charge/discharge. The commands are
checked against installed limits and the carried energy state. These `max`
operations define the saved control rule; they do not saturate commands at
equipment limits. This bounded controller format has no arbitrary callbacks,
forecast access, controller fitting, or general feedback language.

Use `model.as_claim(controller=controller)` or
`model.as_claim().with_controller(controller)`. The binding requires the storage
model's causal permissions and all its service and terminal requirements.
Perfect-foresight and fixed-policy-class audits remain separate benchmarks.
Freezing requires an explicit controller and does not require a passing audit.

## State, paths and evidence

The frozen binding includes the full storage design, tree in caller order,
signal labels, period durations, initial energy, terminal bounds, controller,
causal permissions, units and numerical tolerance. It uses the existing
`frozen_controller/v1` envelope with a nested `storage_controller/v1` declaration.
Both have content identities. `StorageController.to_json()` / `from_json()` also
round-trip the rule independently. Loading validates the reconstructed contract
without executing the controller or invoking an optimizer.

Paths execute in caller order. **Each is an independent two-period experiment
starting at the declared initial energy.** Within a path, the event starts from
the preparation stage's actual calculated state. Terminal bounds are checked at
the end; they do not reset the store. Reordering paths changes their replay order
but does not change their commands or state trajectories. This is finite-tree
replay, not a continuous historical time series or indefinite-operation claim.

`frozen.as_claim(tree=new_tree)` explicitly binds new paths and signal labels to
the same controller and physical settings. New period lengths, if supplied,
become part of that new claim. The original frozen binding remains unchanged.
A bare physical `FiniteSet` cannot replace the tree: it lacks observation timing.
`evaluate_result()` and `audit_result()` evaluate the entire bound tree; point
queries, sensitivities and distance searches are unsupported.

Each evaluation's `replay` evidence contains `details["paths"]`, with ordered
stage observations, commands, state before and after each stage, residuals,
violated constraints and any unresolved error. Quantity arrays follow the
contract's named axes, independently of trace order. Audits retain these traces
and use `fixed_policy_failure` witnesses. Failure concerns that saved rule only;
another controller may still satisfy the same engineering commitment.

An unknown preparation signal without a default stops that path before a
command is issued. Coverage becomes partial and an otherwise passing audit is
inconclusive. A verified violation on another path or an earlier completed check
still establishes failure. Missing states and residuals are not filled with zero.

Calculations use exact rational values of the declared floats, rounding each
command and stage state once to a physical float. Checks use those actual carried
states and commands, with residuals rounded outward and the storage contract's
absolute tolerance. Overflow remains unresolved while preserving completed
checks and issued commands. Standard and compact `process_result/v1` exports
retain this evidence. Controller revisions also work with `compare_changes`.
