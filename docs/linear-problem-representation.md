# Shared production representation for linear problems

Native linear dispatch, local objective sensitivity, equipment relief and
controller synthesis use one internal representation of the declared equations.
This keeps physical row construction consistent as each query changes the
optimization variables or coordinates. Public declarations, result formats and
certificate identities retain their existing meanings.

## Physical declarations

`_linear_compile.compile_model` translates a `LinearProcessAdapter` and its
selected requirements into `_linear_problem.LinearProblem`. Network models and
supported IDAES linear exports enter through that same adapter. The representation
does not compile general nonlinear IDAES models.

The immutable representation contains:

- **Variables:** names, physical units, bounds and explicit `control`, `state` or
  `auxiliary` roles. Renamed scenario variables retain their source names.
- **Input coordinates:** names and units, kept separate from optimization variables.
- **Physical rows:** affine residuals with native-unit tolerances. Each row keeps
  its operating-limit or requirement identity, output, direction, unit and
  positive numerical residual scale.
- **Objectives:** an affine minimization cost, or centered, scaled squared tracking
  terms. Maximization is represented by a signed cost with the original direction
  retained for reporting.

For a declared output `y`, a limit `L`, direction `s` (`1` for an upper limit,
`-1` for a lower limit), tolerance `t` and residual scale `r`, the physical row is
`s * (y - L) <= t`. Numerical row scaling divides **both** sides by `r`; it does
not change the physical tolerance. Arithmetic uses the exact rational values of
the declared floats until a backend needs numerical coefficients.

## Query-specific transformations

Point dispatch calls `LinearProblem.normalized(point, fixed)`. This substitutes
the realization and prescribed actuator commands, eliminates equal-bound
variables, and writes each free variable as `x = lower + span * z`, with
`0 <= z <= 1`. Eliminated variables retain their original bounds as constant
rows, so an out-of-bounds fixed command remains infeasible. States cannot receive
actuator permissions. The objective undergoes the same coordinate transformation,
including all fixed-variable costs.

Sensitivity uses `explicit_rows(fixed)` to retain physical variables, bounds and
fixed-command equations. Keeping these rows explicit allows a bound or input to
be perturbed without accidentally changing a prescribed command. Change rates
come from the physical row coefficients and source metadata. Its numerical
proposal uses the same coordinate-scaling helper as point dispatch.

`_linear_transform` provides three reusable operations on normalized systems:

- `stack_scenarios` replicates dispatch variables into independent blocks and
  returns the variable mappings. Variables and rows retain scenario indices.
- `augment_relief` adds bounded auxiliary variables that relax named equipment
  rows in physical units. One column can affect the same limit across every
  scenario, giving one shared equipment plan. Requirements and variable bounds
  cannot be relief targets.
- `relax_rows` applies specified physical relief amounts to those equipment rows.

Row selection keeps coefficients, right-hand sides and source metadata together
during conflict reduction. Backend candidates and producer-side exact checks
consume the resulting `LinearSystem` and objective `Polynomial`.

Controller synthesis substitutes permitted affine commands into the same physical
rows. Finite synthesis introduces algebraic states per scenario; box synthesis
uses affine support bounds over the entire box. Tracking squares remain centered
in the numerical formulation. The saved controller still requires the existing
separate audit of its rounded execution. Existing coverage limits, including the
restriction on continuous controller synthesis with algebraic states, remain.

## Verification boundary

These private modules are production infrastructure, not a new public model or
artifact schema. They do not import solver backends. SciPy and optional CVXPY
propose candidates; their status alone does not establish a verified result.

The [independent certificate verifier](certificate-verification.md) keeps its own
reconstruction from the exported contract. It does **not** use the production
compiler, normalized system, objective expansion or relief transformations. A
regression test disables those producer paths before checking exported evidence,
so consolidating production cannot make verification reuse the same equations by
accident.
