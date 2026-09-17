# Local uncertainty in operating outputs and margins

Use `IdaesAdapter.propagate_uncertainty(...)` to estimate which uncertain model
parameters drive selected outputs and operating margins. It computes a local
Jacobian `J` and the full response covariance `J @ covariance @ J.T`, preserving
correlation between parameters and between responses. The existing
[joint bootstrap audit](joint-calibration-uncertainty.md) remains the tool for
checking whole parameter vectors across operating cases.

Install `operange[idaes]`. The worked example uses HiGHS; nonlinear models use
the solver configured on their adapter, commonly Ipopt installed separately.

## Fixed actuator commands

Parameters are an **ordered** tuple of `IdaesVariable` bindings to fixed Vars
or mutable Params in the existing model. They must have matching native units
and cannot alias each other, scenario inputs, or actuator controls. Their
current values define the linearization point. Supply the covariance in that
same order, with entry `[i,j]` in `parameter_unit[i] * parameter_unit[j]`.

Alternatively, supply a named `ParameterCovariance`; it is aligned by parameter
identity and units. [Provider-independent parameter data](parameter-data-and-execution.md)
also supports external point estimates and joint samples without a parmest fit.

```python
from operange.integrations.idaes import IdaesVariable
from examples.local_uncertainty import operating_model

adapter = operating_model()
request = dict(
    parameters=(
        IdaesVariable("efficiency", "fs.boiler.efficiency", "1"),
        IdaesVariable("loss", "fs.boiler.loss", "MW"),
    ),
    covariance=((0.01, 0.04), (0.04, 0.25)),
    outputs=("steam", "fuel"),
    margins=("service", "bound:fs.boiler.fuel:upper"),
)
fixed = adapter.propagate_uncertainty(
    **request, fixed_controls={"fuel": 15},
)
assert fixed.execution == "completed"
summary = fixed.statistics
```

The boiler equation is `steam = efficiency * fuel - loss`. With fuel fixed at
15 MW, the steam Jacobian is `(15, -1)`. The supplied covariance gives steam
variance **1.3 MW²**, compared with 2.5 MW² if its off-diagonal terms were
discarded. This reduction comes from correlated efficiency and heat loss.

Every actuator must receive a fixed command. If there are no actuators,
`fixed_controls` can be omitted. The operating inputs come from `realization`,
or the input space's nominal point. `outputs` selects declared output or
actuator names and defaults to all declared outputs. Define named Pyomo
Expressions and bind them as outputs for utility costs, temperatures, or other
derived quantities; no parallel process model is introduced.

Fixed-command derivatives use Pyomo automatic differentiation and implicit
equality-state sensitivities. A square, numerically regular state Jacobian is
required. The row/column-equilibrated condition number must not exceed `1e10`.
Free reconciliation variables, redundant equations, singular state systems,
and unsupported derivative expressions produce an unresolved result. Source
initialization and numerical feasibility checks run before differentiation.

## Ideal reoptimization

```python
optimized = adapter.propagate_uncertainty(
    **request,
    reoptimize=True,
    steps={"efficiency": 1e-4, "loss": 1e-3},
    relative_tolerance=1e-3,
    absolute_tolerance=1e-6,
)
```

This mode retains the model's single active objective and frees its actuator
controls. It is an ideal response that observes the **actual parameter vector**
at every perturbation. For latent parameters it is a comparison benchmark,
not an implementable controller. The example minimizes fuel while meeting
8 MW of steam demand: nominal fuel is 11.25 MW, and its Jacobian is
`(-14.0625, 1.25)`. Steam and its service margin remain locally constant while
fuel demand changes.

The method solves the baseline and four perturbations per parameter:
`-h`, `+h`, `-h/2`, `+h/2`, using explicit native-unit `steps`. It retains every
evaluation, including failures. Each solve must return a candidate with an
optimal or local-optimal termination whose values pass substitution into
unmodified source equations and bounds. This does not certify optimality.

The reported derivative uses the finer central difference. The coarse/fine
and left/right derivatives must agree entry by entry within
`absolute_tolerance + relative_tolerance * max(abs(compared derivatives))`.
The absolute tolerance applies numerically in each derivative's native unit
`response_unit / parameter_unit`; select it for the scales of your study.
These comparisons and thresholds are exported. Small steps can amplify solver
noise; large steps can expose curvature. Disagreement, any failed solve, a
perturbation outside parameter bounds, or a detected active-set change makes
the whole propagation unresolved. No partial Jacobian is presented as complete.

[IDAES's uncertainty propagation toolbox](https://idaes-pse.readthedocs.io/en/stable/explanations/modeling_extensions/uncertainty_propagation/index.html)
also offers optimization sensitivities through the separate `k_aug` executable.
This API currently uses the adapter's existing solver for the checked
perturbation path, and public Pyomo derivatives for fixed commands. It does not
wrap `k_aug`, build a new optimizer, or silently switch derivative methods.

## Margins and parameter importance

`margins` names source inequalities, service requirements, or variable bounds
using their contract constraint IDs. The response name is `margin:<id>`.
Equality residuals cannot be selected as operating margins. Margin is signed
native-unit slack: positive inside the physical limit, zero at the boundary.
Numerical feasibility tolerances are recorded separately from this slack.

`statistics['responses']` includes values, variances, standard deviations,
and parameter contributions. The contribution for parameter `j` is
`J[j] * (covariance @ J)[j]`. These signed allocations sum to response variance,
sharing each cross-covariance equally between its two parameters. They can be
negative and are an accounting convention, not unique causal attributions.
`ranking` orders parameters by absolute contribution. For a margin with
nonzero standard deviation, `margin_in_standard_deviations` divides slack by
its propagated standard deviation. It is **not a failure probability** or a
distance in parameter space.

Active inequalities and bounds are listed under `numerical['active_limits']`.
Fixed-command derivatives describe the ambient equality model; they can point
outside a physical limit. Nonlinear curvature, branch changes, degeneracy and
changing active constraints can invalidate either local approximation.
Completion does not establish differentiability, global uniqueness, robust
feasibility, a probability distribution, or confidence coverage.

## Reuse the full bootstrap covariance

```python
local = bootstrap.propagate_uncertainty(
    adapter,
    parameter_map={
        "fs.boiler.efficiency": "fs.boiler.efficiency",
        "fs.boiler.heat_loss": "fs.boiler.loss",
    },
    fixed_controls={"fuel": 15},
    outputs=("steam", "fuel"),
    margins=("service",),
)
```

All requested draws must be checked. This method uses the full empirical
sample covariance in calibration estimate order and linearizes at the checked
fit, which can differ from the bootstrap mean. Mapping insertion order does not
reorder covariance. Parameter names (including `steps` keys for reoptimization)
remain the original estimate CUIDs. The result records calibration, bootstrap,
and parent-model references. Review
[calibration credibility](calibration-credibility.md) before interpreting an
estimated covariance as useful information.

## Portable results

```python
from operange.integrations.idaes import IdaesUncertaintyResult

restored = IdaesUncertaintyResult.from_json(fixed.to_json())
summary = restored.statistics
```

`idaes_local_uncertainty/v1` retains parameter order, native units, nominal
values, covariance, response order, Jacobian, baseline and perturbation source
checks, diagnostics, method and scope. Reading and summarizing requires only
base Operange dependencies. Loading validates record consistency and content
identity; it does not independently verify nonlinear mathematics. Statistics
are unavailable when `execution == 'unresolved'`.

Run the complete example:

```bash
uv run --extra idaes python -m examples.local_uncertainty \
    --fixed-output /tmp/fixed-local.json \
    --reoptimized-output /tmp/reoptimized-local.json
```
