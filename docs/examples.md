# Worked examples

Start with the [shared-steam tutorial](index.md), then choose a study below.
All examples use synthetic engineering data and state the scope of their claims.

| Study | What it demonstrates | Guide |
| --- | --- | --- |
| Shared steam header and boiler | Combined loads can exceed shared capacity even when either individual peak fits; inspect a witness, sensitivities and an equipment revision. | [Getting started](index.md) |
| Two adjustable boilers | Redispatch handles individual load peaks; a combined peak requires more shared fuel capacity. | [Linear dispatch](linear-dispatch.md) |
| Distance to a limit | Find the nearest affine requirement boundary or a specified positive violation, with checked distance bounds. | [Failure distance](failure-distance.md) |
| Comparing engineering changes | Distinguish restored service from changes to the uncertainty envelope or service requirement. | [Engineering changes](engineering-changes.md) |
| Frozen boiler allocation | A 50/50 allocation violates the fuel limit; a 60/40 rule and the adjustable benchmark pass. Save and replay the rule. | [Frozen controllers](frozen-controllers.md) |
| Causal storage replay | Preserve preparation signals, carried energy and terminal checks when a saved rule faces held-out paths. | [Storage replay](storage-replay.md) |
| Compressor starts | Staggering fixed starts reduces the peak from 700 kVA to 480 kVA against a 500 kVA limit over the same 65 scenarios. | [Startup loads](startup.md) |
| Heat integration | Combined stream-flow changes increase the heating target from 10 kW to 100 kW, exceeding a 50 kW utility limit. | [Heat-cascade targets](pinch.md) |

The [capability table](api.md#what-the-built-in-adapters-establish) gives the
supported questions and limits for each model. For example, a heat-cascade
target screen does not establish installed exchanger-network feasibility, and
two-period storage replay does not establish indefinite operation.

The experimental [DWSIM connection example](dwsim.md) uses the installed
simulator's bundled Humid Air flowsheet to exchange input batches and checked
stream results with Python. It requires DWSIM's Script Manager and is not yet
a public model adapter.

## Run examples from the checkout

After [setting up the environment](installation.md#install-from-a-checkout), run
any of these commands from the repository root:

```bash
uv run python -m examples.steam_header
uv run python -m examples.linear_dispatch
uv run python -m examples.failure_distance
uv run python -m examples.engineering_changes
uv run python -m examples.frozen_controllers
uv run python -m examples.storage_replay
uv run python -m examples.startup
uv run python -m examples.pinch
```

The heat-recovery and thermal-storage reference studies also include downstream
reporting and priced repair catalogs:

```bash
uv run python -m examples.heat_recovery
uv run python -m examples.thermal_storage
```

Browse the [example source files](https://github.com/nahuaque/operange/tree/main/examples)
for complete declarations. They are repository consumers of the installed API;
the wheel contains the library, not the examples or their reporting layer.
Core comparisons retain engineering audits in caller order and do not select
an investment.
