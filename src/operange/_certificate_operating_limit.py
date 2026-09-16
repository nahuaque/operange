"""Check range nesting and bound certificates without rerunning the search."""

from dataclasses import replace
from fractions import Fraction as F

from ._certificate_math import Context, UnsupportedCertificate, require
from .contract_types import plain
from .primitives import BoxSet, Parameter, finite


def verify_operating_limit(verifier, result):
    def declarations():
        if result.execution in ("unsupported", "invalid"):
            raise UnsupportedCertificate(
                "This operating-limit query establishes no bounds."
            )
        ctx = Context(result.contract)
        request = result.request
        require(
            set(request)
            == {
                "query",
                "parameter",
                "tolerance",
                "max_iterations",
                "max_vertices",
                "backend",
                "range_semantics",
            },
            "unexpected operating-limit query fields",
        )
        require(
            request["query"] == "operating_limit"
            and request["range_semantics"]
            == "all values from the original lower endpoint through the candidate endpoint",
            "unsupported operating-range semantics",
        )
        require(type(ctx.domain) is BoxSet, "operating range requires a box")
        p = next(
            (p for p in ctx.domain.parameters if p.name == request["parameter"]), None
        )
        require(type(p) is Parameter, "operating coordinate must be a declared scalar")
        require(
            ctx.model.measurements is None
            or p.name not in ctx.model.measurements.error_inputs,
            "operating coordinate is a sensor error",
        )
        require(finite(request["tolerance"], "tolerance") > 0, "invalid limit tolerance")
        require(
            type(request["max_iterations"]) is int and request["max_iterations"] >= 0,
            "invalid iteration budget",
        )
        require(
            type(request["max_vertices"]) is int and request["max_vertices"] > 0,
            "invalid vertex budget",
        )
        require(request["backend"] in ("scipy", "cvxpy"), "unknown backend")
        require(
            len(result.trials) <= request["max_iterations"] + 2,
            "trial count exceeds the query budget",
        )
        require(
            len({t.endpoint for t in result.trials}) == len(result.trials),
            "duplicate operating-range trial",
        )
        for t in result.trials:
            require(
                p.lower <= t.endpoint <= p.upper,
                "trial endpoint is outside the declared search range",
            )
            # Reconstruct only the allowed domain change. Equipment, service,
            # measurements, permissions and the saved controller must be identical.
            domain = plain(result.contract.domain)
            selected = next(
                item for item in domain["parameters"] if item["name"] == p.name
            )
            selected.update(upper=t.endpoint, nominal=min(p.nominal, t.endpoint))
            require(
                t.audit.contract == replace(result.contract, domain=domain),
                "trial changes assumptions beyond the declared operating endpoint",
            )
        if result.lower is None and result.upper is None:
            raise UnsupportedCertificate("No bound certificate is present.")
        if result.resolution == "limit_bracketed":
            require(
                F(result.upper) - F(result.lower) <= F(request["tolerance"]),
                "operating-limit gap exceeds tolerance",
            )
        return ctx

    ctx = verifier.check("operating_limit/declarations", declarations)
    if ctx is None:
        return
    for i, trial in enumerate(result.trials):
        path = f"operating_limit/trials/{i}"
        if trial.outcome == "pass":
            verifier.result(trial.audit, path + "/achievable_audit")
        elif trial.outcome == "fail":
            # A single impossible realization excludes a whole prefix. Complete
            # coverage and a successful enclosure are unnecessary for this side.
            def failure(trial=trial):
                witness = trial.audit.payload.witness
                children = {e.result_id: e for e in trial.audit.supporting_evaluations}
                child = children[witness.details["evaluation_ref"]["artifact_id"]]
                require(
                    child.contract == trial.audit.contract,
                    "failure has another contract",
                )
                require(
                    witness.details["evaluation_ref"] == child.ref.to_dict()
                    and witness.realizations == (child.request["realization"],)
                    and child.payload.feasibility == "infeasible",
                    "excluded endpoint lacks a matching infeasible realization",
                )
                expected_kind = (
                    "fixed_policy_failure"
                    if ctx.controller or ctx.policy.mode == "fixed"
                    else "individual_infeasibility"
                )
                require(
                    witness.kind == expected_kind,
                    "failure changes the operating-policy scope",
                )
                affected = (
                    tuple(
                        c.constraint_ref
                        for c in child.payload.constraint_checks
                        if c.assessment == "violated"
                    )
                    if ctx.controller
                    else tuple(
                        c.constraint_id for c in ctx.included if c.kind != "equation"
                    )
                )
                require(
                    witness.affected_constraints == affected,
                    "limiting conditions differ from the checked failure",
                )
                return child

            child = verifier.check(path + "/limiting_witness", failure)
            if child is not None:
                verifier.result(child, path + "/infeasibility")
        # Unresolved trials assert no mathematical bound. They are preserved as
        # diagnostics, and cannot determine either public endpoint.
