# Startup spikes and fixed start schedules

The startup slice evaluates supplied load profiles and checks whether fixed
start schedules respect a shared capacity. It retains the peak, its timing and
its duration instead of replacing a transient with a time-averaged load.

| Declaration | Meaning |
| --- | --- |
| `PiecewiseLinearProfile` | Values at elapsed seconds with explicit linear interpolation between knots; zero before time zero and the final value held afterwards |
| `reference.StartupEvent` | One named off → starting → running transition with a supplied profile and a fixed scheduled command time |
| `reference.StartupLoadAdapter` | Sum scalar load envelopes with a constant background; evaluate peaks and time integrals; audit a finite uncertainty set |

Profile provenance identifies the supplied data or model. The example uses
synthetic kVA envelopes. This additive scalar-load model does not infer motor
acceleration, network voltage, phase relationships, reactive power, pressure
dynamics or protection behaviour from a compressor rating.

## Commands and uncertainty

Each event declares three uncertain inputs. `amplitude_scale` multiplies the
whole profile, including the final running value. `duration_scale` multiplies
its elapsed times. Both scales must be positive. `timing_jitter_seconds` is an
offset added to the scheduled command time. Use `event.input_name(field)` to
obtain a coordinate name, and supply `Scenario` values in a `FiniteSet` over
`adapter.input_space`.

The command time is a fixed control in `RecoursePolicy`. Changing it changes
the operating contract while preserving the uncertainty domain and equipment.
Before the actual start, event load is zero. The transformed profile then
describes startup, with an optional initial jump. After the final knot its
running value is held. There is one start per event.

Every event must start at or after zero and finish starting by the horizon.
An out-of-window realization is rejected, so delaying a startup out of view
cannot produce a passing audit.

## Two compressors on one supply

```{literalinclude} ../examples/startup.py
:language: python
:start-at: from dataclasses import replace
:end-before: def run_example():
```

The profile rises to 300 kVA at 0.1 seconds, falls to 250 kVA at one second,
and settles at 100 kVA after two seconds. Both compressors initially have
command times of 0.2 seconds. A 40 kVA background shares a 500 kVA limit over
an eight-second horizon.

The uncertainty set contains the nominal case and 64 combinations of amplitudes
`0.9/1.1`, duration scales `0.8/1.25` and timing jitter `-0.1/+0.1` seconds for
the two events.

| Schedule | Nominal peak | Largest peak across the 65 scenarios | Verdict |
| --- | --- | --- | --- |
| Both commanded at 0.2 s | 640 kVA | 700 kVA | Fail |
| Commands at 0.2 s and 3.1 s | 440 kVA | 480 kVA | Pass |

Staggering retains both events, the same uncertainty set, horizon and capacity.
It is an operating change; the example does not claim an optimal schedule or
attach an economic value.

Run from the repository root to export `process_result/v1` JSON:

```bash
uv run python -m examples.startup
```

## Coverage between samples

For each scenario, the adapter shifts and stretches the profiles, collects all
resulting breakpoints and the horizon endpoints, and checks the aggregate at
those points. It retains both sides of initial jumps. Between consecutive
breakpoints the aggregate is linear, so endpoint values bound its maximum.
Integration uses the trapezoidal formula on each linear segment, without
integrating across a jump.

Calculations use exact rational arithmetic on the declared floating-point
inputs. Nearby event times are not collapsed by a large time shift. Evidence
retains exact time strings alongside floating values. Exported residuals round
towards positive infinity before comparison with the declared tolerance.

This establishes **continuous-time coverage of the declared interpolated
profiles**. It does not establish that sampled measurements have no unmeasured
spikes. Linear interpolation of imported measurements is a modelling assumption
about the waveform.

Uncertainty coverage is **complete finite enumeration**. These 65 scenarios do
not certify intermediate amplitudes, durations or timing offsets. Intermediate
timing can create overlap absent at the extremes. Both the time and uncertainty
coverage are explicit in the evidence.

## Results and requirements

Use `evaluate_result` and `audit_result` through the common `Claim` interface.
Evaluations report `peak_load`, `peak_time`, `integrated_load`, constraint
residuals and fixed controls. Profile evidence retains actual start and
completion times, breakpoints, individual loads and the aggregate.

`shared_capacity` compares maximum aggregate load with `capacity`, using
`tolerance` in the profile unit. Optional `integral_limit` adds
`startup_exposure`, comparing the time integral over the whole horizon with
that limit, using `integral_tolerance`. The integral includes background load
and running tails.

Integral units are load-unit times seconds: A·s for current, not I²t; kVA·s for
apparent load, not active electrical energy. No thermal or protection model is
inferred. Equal peaks can still have different integrals because their durations
differ.

A failed audit returns a `fixed_policy_failure` witness carrying the uncertain
realization, affected requirements and profile evidence. Re-evaluating that
realization under the same claim reproduces the failure. It refutes the fixed
schedule, not every possible start sequence.

Invalid profiles, truncated events and unresolved calculations cannot establish
a pass. A verified failure can still refute a claim when another scenario is
unresolved; coverage then remains partial. Unsupported operations never silently
fall back to sampling.

## Scope of this slice

There is no general `TrajectorySet` or hybrid process simulator yet. This slice
uses supplied piecewise-linear profiles, fixed schedules and the existing finite
uncertainty set. Continuous uncertainty optimization, adaptive scheduling,
repeated starts with shared state, initial pressure or speed dynamics,
derivatives, nearest-failure searches and controller synthesis remain future
work.

An external simulator can supply profiles now. Executing its dynamics and
verifying its own time discretization require a separate adapter and evidence.
