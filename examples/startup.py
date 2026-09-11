"""Two supplied compressor startup envelopes sharing a scalar electrical limit."""

from dataclasses import replace
from itertools import product
import json

from operange import FiniteSet, PiecewiseLinearProfile, Scenario
from operange.reference import StartupEvent, StartupLoadAdapter


def example():
    # Synthetic kVA envelopes, not motor equations or network/protection analysis.
    profile = PiecewiseLinearProfile(
        (0, 0.1, 1, 2),
        (0, 300, 250, 100),
        "kVA",
        "Synthetic compressor start: linear between knots; final running load held",
    )
    model = StartupLoadAdapter(
        "Two compressor starts on a shared supply",
        tuple(
            StartupEvent(name, profile, 0.2, "Synthetic fixed start command")
            for name in ("compressor_a", "compressor_b")
        ),
        horizon_seconds=8,
        capacity=500,
        background_load=40,
    )
    scenarios = [
        Scenario("nominal", model.input_space.nominal, "Synthetic nominal start")
    ]
    # Exhaustive coverage of these 64 combinations, not of the intervals between them.
    choices = tuple(product((0.9, 1.1), (0.8, 1.25), (-0.1, 0.1)))
    for index, pair in enumerate(product(choices, repeat=2)):
        values = {
            event.input_name(field): value
            for event, settings in zip(model.events, pair)
            for field, value in zip(
                ("amplitude_scale", "duration_scale", "timing_jitter_seconds"), settings
            )
        }
        scenarios.append(
            Scenario(
                f"combination_{index:02d}",
                values,
                "Synthetic finite stress combination",
            )
        )
    domain = FiniteSet(model.input_space, tuple(scenarios))
    staggered = replace(
        model,
        events=(
            model.events[0],
            replace(model.events[1], scheduled_start_seconds=3.1),
        ),
    )
    return model, staggered, domain


def run_example():
    simultaneous, staggered, domain = example()
    nominal = simultaneous.as_claim(domain).evaluate_result(
        domain.scenario("nominal").values
    )
    results = {
        "nominal": nominal,
        "simultaneous": simultaneous.as_claim(domain).audit_result(),
        "staggered": staggered.as_claim(domain).audit_result(),
    }
    return {name: result.to_dict() for name, result in results.items()}


def main():
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
