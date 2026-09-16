# Controllers with uncertain measurements

`MeasurementModel` separates the readings available to a controller from the
true process conditions used to check service and equipment limits. Attach it
to `LinearProcessAdapter(measurements=...)`, then bind an explicit controller or
use `claim.synthesize_controller()` to choose its coefficients.

## Declare truth, error and readings

The uncertainty domain contains both physical inputs and latent sensor errors.
For example, true steam demand spans 8–10 MW and the meter bias spans
−0.25–0.25 MW. A reading is an affine function of these joint inputs:

```python
from operange import AffineMeasurement, AffineTerm, MeasurementModel

measurements = MeasurementModel(
    "Steam demand meter",
    readings=(
        AffineMeasurement(
            "measured_load", "MW", 0,
            (AffineTerm("load", 1, "MW/MW"),
             AffineTerm("meter_bias", 1, "MW/MW")),
            "Measured demand equals true demand plus unknown bias",
        ),
    ),
    error_inputs=("meter_bias",),
)
```

`load` and `meter_bias` must both be declared in the model's input space and
domain. Physical output equations use the true load. Inputs listed in
`error_inputs` are prohibited from appearing in physical output equations.
Reading names must be distinct from input, control and output names. Coefficient
units must be written as `reading_unit/input_unit`; unit labels do not perform
conversion.

With a measurement model, controller permissions can name only its readings.
The default permissions expose all declared readings; explicit `DecisionRule`
permissions can restrict each control further. Raw truth and bias coordinates
are unavailable to commands. To expose an exactly observed physical input,
declare a separate identity reading for it. Multiple terms can represent an
aggregate meter, and several readings can share the same error coordinate to
represent common calibration bias. Separate error coordinates in a box allow
their errors to vary independently; finite cases and hulls can declare other
joint support assumptions.

## Synthesize and test a steam rule

The complete example declares delivery minus true demand as `steam_balance`,
requires that balance to be nonnegative, caps excess steam at 0.5 MW, and
minimizes worst-case fuel use at 80% boiler efficiency:

```{literalinclude} ../examples/measurement_uncertainty.py
:language: python
:start-at: def example():
:end-before: def run_example():
```

Following the meter directly fails when it underreads. A suitable affine rule
is approximately `steam = measured_load + 0.25 MW`. Across the joint domain it
delivers between the true demand and 0.5 MW above that demand, with worst-case
fuel use approximately 13.125 MW. Synthesis may use the declared numerical
tolerances and need not return exactly these coefficients.

```python
from operange import FrozenController, verify_result

search = example().synthesize_controller(name="Allow for meter error")
if search.verified:
    frozen = FrozenController.from_json(search.freeze().to_json())
    replay = frozen.evaluate_result({"load": 9, "meter_bias": -0.25})
    verification = verify_result(search.audit)
```

Replay receives the complete true/error realization. It computes the readings,
executes the saved rule, then checks the physical response against truth.
Missing errors are invalid inputs; they are never silently set to zero.
For execution with readings already available, use
`search.controller.commands({"measured_load": 8.75})`. This computes commands
only; a physical feasibility claim still needs the joint realization.

The frozen artifact includes the measurement declaration, domain, permissions
and controller. Rebinding the same rule to a wider error domain explicitly tests
new assumptions without retuning. In this example, widening the bias to
±0.5 MW causes the original controller to fail.

```bash
uv run --extra cvxpy python -m examples.measurement_uncertainty
```

## Numerical evidence and scope

Each reading is computed as an exact rational affine sum of the declared
floats, then rounded once to a sensor float. Each command consumes those sensor
floats, forms its own exact affine sum, and rounds once to a command float.
The physical residual check uses that actual command. No clipping is applied.
Execution evidence records the readings and each control's observations.

Continuous audits substitute the measurement maps into the controller while
bounding both rounding steps. They preserve joint-input correlations, including
shared bias, before bounding physical residuals. The enclosure also records
each sensor's range and rounding allowance. Independent certificate verification
recomputes these quantities and actual readings without executing the production
measurement or controller implementation. Its existing box, finite-case and
hull scope still applies; see [certificate verification](certificate-verification.md).

Synthesis has the existing [domain and objective scope](controller-synthesis.md).
Linear objectives over boxes retain the scalable support formulation, including
aggregate readings and shared errors. Numerical proposals are accepted only
after the actual rounded controller passes the full audit and performance
checks. Conservative rounding bounds can leave a search unresolved. The physical
audit is independently verifiable; the synthesis report does not certify an
optimum across all controllers.

An adjustable measurement-based claim requires an explicit or synthesized
controller before point evaluation or auditing: independently redispatching at
each true realization would bypass its information restrictions. Fixed commands
remain evaluable. All declared readings must remain finite for controller
execution; overflow leaves the response unresolved.

These declarations describe static, bounded measurement uncertainty. Bounds,
dependencies and provenance are caller-supplied assumptions. They introduce no
probabilities, calibration fitting, sensor dynamics, observation delays, or
claim that a bias persists across a time series. Multiplying an unknown gain
error by a varying true input is bilinear and is outside this affine interface.
