"""Quadratic process experiment and bounded cold/reuse microbenchmark.

Use PYTHONPATH=../residopt/src and PYTHONDONTWRITEBYTECODE=1 to run against
the sibling source without installing it or changing either repository.
"""

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter
import warnings

import numpy as np
import scipy

from operange import BoxSet, EllipsoidSet, Parameter
from operange.experimental._quadratic_bounds import PreparedTrustRegion
from operange.experimental._residopt_quadratic import (
    PreparedResidualSDP,
    backend_identity,
    load_backend,
)
from operange.experimental.quadratic import (
    QuadraticModel,
    QuadraticProcessAdapter,
)


def example():
    space = BoxSet(
        (
            Parameter("flow", "kg/s", 100, 80, 120, 10, "Synthetic flow scale"),
            Parameter(
                "temperature", "degC", 50, 40, 60, 5, "Synthetic temperature scale"
            ),
        )
    ).space
    domain = EllipsoidSet(
        space,
        {
            "flow": {"flow": 1, "temperature": 0.4},
            "temperature": {"flow": 0.4, "temperature": 1},
        },
        radius=1.5,
    )
    model = QuadraticModel(
        "Synthetic pressure-loss response",
        space,
        "kPa",
        "pressure",
        25,
        {"flow": 3, "temperature": -1},
        {
            "flow": {"flow": 4, "temperature": 1},
            "temperature": {"flow": 1, "temperature": 2},
        },
        50,
        "Synthetic quadratic only; no plant validation or surrogate-error bound",
        tolerance=1e-5,
    )
    return model, domain


def engineering_results(solver):
    model, domain = example()
    results = {}
    for backend in ("trust_region", "residopt_sdp"):
        for label, limit in (("passing", 50), ("failing", 27)):
            claim = QuadraticProcessAdapter(
                replace(model, limit=limit), backend, solver
            ).as_claim(domain)
            results[f"{backend}_{label}"] = claim.audit_result().to_dict()
    return results


def workload(H, directions, references, *, backend, reuse, solver, tolerance):
    build_seconds, solve_seconds, errors, gaps = 0.0, [], [], []
    prepared = None
    for b, reference in zip(directions, references):
        started = perf_counter()
        try:
            if prepared is None or not reuse:
                prepared = (
                    PreparedTrustRegion(H)
                    if backend == "trust_region"
                    else PreparedResidualSDP(H, solver=solver)
                )
                build_seconds += perf_counter() - started
            started = perf_counter()
            if backend == "trust_region":
                upper = prepared.maximize(b, tolerance=tolerance).upper
            else:
                upper, _ = prepared.upper_bound(b, tolerance=tolerance)
            solve_seconds.append(perf_counter() - started)
            if upper < reference.lower - tolerance:
                errors.append("bound below independent feasible value")
            gaps.append(upper - reference.lower)
        except (RuntimeError, ValueError) as exc:
            solve_seconds.append(perf_counter() - started)
            errors.append(f"{type(exc).__name__}: {exc}")
            prepared = None
    return {
        "backend": backend,
        "reuse": reuse,
        "build_seconds": build_seconds,
        "solve_seconds": sum(solve_seconds),
        "total_seconds": build_seconds + sum(solve_seconds),
        "median_solve_seconds": median(solve_seconds),
        "maximum_gap": max(gaps, default=None),
        "successful_calls": len(directions) - len(errors),
        "calls": len(directions),
        "errors": errors,
    }


def benchmark(*, solver, dimensions, repeats, size, tolerance):
    trials, fixtures = [], []
    for dimension in dimensions:
        rng = np.random.default_rng(20260911 + dimension)
        Q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        H = Q @ np.diag(np.linspace(-2, 4, dimension)) @ Q.T
        H = 0.5 * (H + H.T)
        spectral_error = float(
            np.max(np.abs(np.linalg.eigvalsh(H) - np.linspace(-2, 4, dimension)))
        )
        if not np.all(np.isfinite(H)) or spectral_error > 1e-10:
            raise RuntimeError("Benchmark fixture failed its spectral check")
        fixtures.append({"dimension": dimension, "spectral_error": spectral_error})
        directions = rng.normal(size=(size, dimension))
        reference = PreparedTrustRegion(H)
        references = [reference.maximize(b, tolerance=tolerance) for b in directions]
        for repeat in range(repeats):
            backends = (
                ("trust_region", "residopt_sdp")
                if repeat % 2 == 0
                else ("residopt_sdp", "trust_region")
            )
            for backend in backends:
                for reuse in (False, True):
                    trials.append(
                        {
                            "dimension": dimension,
                            "repeat": repeat,
                            **workload(
                                H,
                                directions,
                                references,
                                backend=backend,
                                reuse=reuse,
                                solver=solver,
                                tolerance=tolerance,
                            ),
                        }
                    )
    summary = []
    for dimension in dimensions:
        for backend in ("trust_region", "residopt_sdp"):
            for reuse in (False, True):
                rows = [
                    r
                    for r in trials
                    if r["dimension"] == dimension
                    and r["backend"] == backend
                    and r["reuse"] == reuse
                ]
                summary.append(
                    {
                        "dimension": dimension,
                        "backend": backend,
                        "reuse": reuse,
                        "median_build_seconds": median(
                            r["build_seconds"] for r in rows
                        ),
                        "median_total_seconds": median(
                            r["total_seconds"] for r in rows
                        ),
                        "median_solve_seconds": median(
                            r["median_solve_seconds"] for r in rows
                        ),
                        "maximum_gap": max(
                            (
                                r["maximum_gap"]
                                for r in rows
                                if r["maximum_gap"] is not None
                            ),
                            default=None,
                        ),
                        "successful_calls": sum(r["successful_calls"] for r in rows),
                        "calls": sum(r["calls"] for r in rows),
                    }
                )
    return {
        "methodology": {
            "workload": "fixed dense indefinite Hessian; varying linear terms on a unit ball",
            "size": size,
            "repeats": repeats,
            "dimensions": dimensions,
            "bound_tolerance": tolerance,
            "timed": "backend construction and verified solve calls; first solve includes canonicalization",
            "excluded": "imports, normalization, common independent cross-check and result serialization",
            "not_measured": "hybrid strategy selection, causal recourse, multi-constraint design optimization",
        },
        "summary": summary,
        "fixture_checks": fixtures,
        "trials": trials,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workload-size", type=int, default=12)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.workload_size < 1
        or any(d < 1 for d in args.dimensions)
    ):
        parser.error("workload sizes, dimensions and repeats must be positive")
    residopt, cp = load_backend()
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        engineering = engineering_results(args.solver)
        measurements = benchmark(
            solver=args.solver,
            dimensions=args.dimensions,
            repeats=args.repeats,
            size=args.workload_size,
            tolerance=1e-6,
        )
    counts = Counter(
        (w.category.__name__, str(w.message), w.filename, w.lineno) for w in observed
    )
    report = {
        "source_revision": args.source_revision,
        "backend_identity": backend_identity(residopt, cp),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "cvxpy": cp.__version__,
            "solver": args.solver,
        },
        "engineering_results": engineering,
        "benchmark": measurements,
        "observed_warnings": [
            dict(category=c, message=m, file=f, line=line, count=count)
            for (c, m, f, line), count in counts.items()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(f"Wrote {args.output}")
    print(f"Recorded {len(observed)} warnings; see observed_warnings in the report")
    for name, result in report["engineering_results"].items():
        print(name, result["payload"]["verdict"])
    for row in report["benchmark"]["summary"]:
        print(row)


if __name__ == "__main__":
    main()
