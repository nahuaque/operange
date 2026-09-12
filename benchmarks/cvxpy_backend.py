"""Reproducible full-query timings; these are measurements, not performance tests.

Run from the checkout: uv run --extra cvxpy python -m benchmarks.cvxpy_backend
The first CVXPY query includes its lazy import. Warm timings include all physical
and evidence checks. A new process gives the cleanest cold measurement.
"""

import argparse
import json
import platform
from statistics import median
from time import perf_counter

import scipy

from operange._cvxpy_backend import clear_cache, load_cvxpy
from examples.convex_backends import intersection_example
from examples.linear_dispatch import example


def measure(query, repeats):
    clear_cache()
    start = perf_counter()
    first = query(0)
    cold = perf_counter() - start
    elapsed = []
    for i in range(repeats):
        start = perf_counter()
        query(i)
        elapsed.append(perf_counter() - start)
    return {
        "cold_seconds": cold,
        "warm_median_seconds": median(elapsed),
        "first_outcome": first,
    }


def run(repeats):
    model, domain = example()
    claim = model.as_claim(domain)

    def dispatch(backend, i):
        # Alternating loads changes RHS values while reusing the row matrix.
        scenario = domain.scenarios[i % len(domain.scenarios)]
        result = claim.evaluate_result(scenario.values, backend=backend)
        expected = "infeasible" if scenario.name == "combined" else "feasible"
        if result.payload.feasibility != expected:
            raise RuntimeError(f"unexpected dispatch result: {result.execution}")
        return result.payload.feasibility

    measurements = {
        backend: measure(lambda i, backend=backend: dispatch(backend, i), repeats)
        for backend in ("scipy", "cvxpy")
    }
    _, joint = intersection_example()

    def support(i):
        weights = {"dryer": 1, "evaporator": 1 if i % 2 else -1}
        result = joint.maximize_linear(weights)
        if result.lower is None or result.upper is None:
            raise RuntimeError(f"unresolved support: {result.to_dict()}")
        return result.status

    measurements["intersection"] = measure(support, repeats)
    return {
        "python": platform.python_version(),
        "scipy": scipy.__version__,
        "cvxpy": load_cvxpy().__version__,
        "repeats": repeats,
        "scope": "two controls, three rows; full checked query timings",
        "measurements": measurements,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    print(json.dumps(run(args.repeats), indent=2, allow_nan=False))
