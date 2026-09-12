"""Convex affine-policy proposals and exact checks of their rounded execution."""

from dataclasses import replace
from fractions import Fraction as F
from itertools import product

from ._controller_envelope import ControllerEnvelope
from ._cvxpy_backend import BackendUnavailable, load_cvxpy, numeric_vector
from ._numeric import exact_dot, round_down, round_up
from .affine import AffineTerm
from .contract_types import Diagnostic, Evidence, nonempty
from .controller_synthesis import ControllerSynthesis
from .controllers import AffineControlRule, AffineController, _ControllerAdapter
from .convex_hull import ConvexHullSet
from .domains import FiniteSet, Scenario
from .linear_process import LinearProcessAdapter
from .objectives import ControlTrackingObjective, LinearObjective
from .primitives import BoxSet, finite


def _generators(domain, limit):
    if type(domain) is BoxSet:
        parameters = sorted(domain.scalar_parameters, key=lambda p: p.name)
        count = 2 ** sum(p.lower != p.upper for p in parameters)
    elif type(domain) is FiniteSet:
        count = len(domain.scenarios)
    elif type(domain) is ConvexHullSet:
        count = len(domain.vertices)
    else:
        raise NotImplementedError(
            "Synthesis supports built-in boxes, finite sets and explicit convex hulls."
        )
    if count > limit:
        raise NotImplementedError(
            f"Complete coverage needs {count} generators, exceeding max_vertices={limit}; none were solved."
        )
    if type(domain) is BoxSet:
        return tuple(
            Scenario(
                f"corner_{i}",
                dict(zip((p.name for p in parameters), values)),
                "Complete synthesis box corners",
            )
            for i, values in enumerate(
                product(
                    *(
                        (p.lower,) if p.lower == p.upper else (p.lower, p.upper)
                        for p in parameters
                    )
                )
            )
        )
    return domain.scenarios if type(domain) is FiniteSet else domain.vertices


def _validate(claim, objective, name, max_vertices):
    model = (
        claim.adapter.model
        if type(claim.adapter) is _ControllerAdapter
        else claim.adapter
    )
    if type(model) is not LinearProcessAdapter:
        raise NotImplementedError(
            "Affine synthesis requires the declarative LinearProcessAdapter."
        )
    if not isinstance(name, str):
        raise ValueError("controller name must be a string")
    nonempty(name, "controller name")
    if type(max_vertices) is not int or max_vertices < 1:
        raise ValueError("max_vertices must be a positive integer")
    objective = model.objective if objective is None else objective
    if type(objective) not in (LinearObjective, ControlTrackingObjective):
        raise ValueError(
            "Declare a LinearObjective or ControlTrackingObjective for synthesis."
        )
    replace(
        model, objective=objective
    )  # Reuse the model's output and target-unit validation.
    dummy = AffineController(
        name,
        tuple(
            AffineControlRule(
                c.name,
                c.unit,
                claim.recourse.rule(c.name).fixed_value
                if claim.recourse.rule(c.name).fixed_value is not None
                else c.lower,
            )
            for c in model.controls
        ),
        "Synthesis permission validation",
    )
    binding = claim.with_controller(dummy)
    if not binding.capabilities.audit.supported:
        raise NotImplementedError(binding.capabilities.audit.scope)
    scenarios = _generators(claim.domain, max_vertices)
    for scenario in scenarios:
        if claim.domain.membership(scenario.values).status != "inside":
            raise ValueError(
                "Every synthesis generator must have verified domain membership."
            )
    return model, objective, scenarios


class PolicyProgram:
    def __init__(self, cp, claim, model, objective, scenarios):
        self.cp, self.model, self.objective = cp, model, objective
        self.scenarios = scenarios
        self.ranges = {
            n: (
                min(F(s.values[n]) for s in scenarios),
                max(F(s.values[n]) for s in scenarios),
            )
            for n in claim.domain.space.names
        }
        self.rules = {}
        for control in model.controls:
            permission = claim.recourse.rule(control.name)
            fixed = permission.fixed_value
            if fixed is None and control.lower == control.upper:
                fixed = control.lower
            observes = (
                tuple(
                    sorted(
                        n
                        for n in permission.observes
                        if self.ranges[n][0] != self.ranges[n][1]
                    )
                )
                if fixed is None
                else ()
            )
            variable = (
                cp.Variable(1 + len(observes), name=f"rule_{control.name}")
                if fixed is None
                else None
            )
            self.rules[control.name] = (control, fixed, observes, variable)

    def commands(self, point):
        commands = {}
        for n, (control, fixed, observes, variable) in self.rules.items():
            if fixed is not None:
                commands[n] = self.cp.Constant(fixed)
            else:
                features = [1.0] + [
                    float(
                        (F(point[k]) - self.ranges[k][0])
                        / (self.ranges[k][1] - self.ranges[k][0])
                    )
                    for k in observes
                ]
                commands[n] = control.lower + float(
                    F(control.upper) - F(control.lower)
                ) * (features @ variable)
        return commands

    def output(self, output, point, commands):
        constant = F(output.offset) + sum(
            (
                F(t.coefficient) * F(point[t.variable])
                for t in output.terms
                if t.variable not in commands
            ),
            F(0),
        )
        return self.cp.Constant(finite(float(constant), "output constant")) + sum(
            (
                t.coefficient * commands[t.variable]
                for t in output.terms
                if t.variable in commands
            ),
            0,
        )

    def solve(self, requirements, reserve):
        cp, model, spec = self.cp, self.model, self.objective
        constraints, epigraph = [], cp.Variable(name="worst_case_cost")
        for scenario in self.scenarios:
            point = scenario.values
            commands = self.commands(point)
            for control in model.controls:
                constraints.extend(
                    (
                        commands[control.name] >= control.lower,
                        commands[control.name] <= control.upper,
                    )
                )
            for row in model.operating_limits + tuple(
                r for r in model.requirements if r.name in requirements
            ):
                residual = row.sign * (
                    self.output(model.output(row.output), point, commands) - row.limit
                )
                constraints.append(
                    residual / row.residual_scale
                    <= (1 - reserve) * row.tolerance / row.residual_scale
                )
            if type(spec) is LinearObjective:
                cost = (1 if spec.sense == "minimize" else -1) * self.output(
                    model.output(spec.output), point, commands
                )
            else:
                cost = sum(
                    t.weight * cp.square((commands[t.control] - t.target) / t.scale)
                    for t in spec.targets
                )
            constraints.append(cost <= epigraph)
        problem = cp.Problem(cp.Minimize(epigraph), constraints)
        report = {
            "backend": f"CVXPY {cp.__version__}",
            "status": "unresolved",
            "reserved_tolerance_fraction": reserve,
        }
        try:
            if type(spec) is LinearObjective:
                problem.solve(
                    solver="SCIPY",
                    scipy_options={
                        "method": "highs",
                        "primal_feasibility_tolerance": model.solver_tolerance,
                        "dual_feasibility_tolerance": model.solver_tolerance,
                    },
                )
            else:
                problem.solve(
                    solver="CLARABEL",
                    tol_gap_abs=model.solver_tolerance,
                    tol_gap_rel=0.0,
                    tol_feas=model.solver_tolerance,
                )
            report["status"] = str(problem.status)
            if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return None, report
            values = {
                n: numeric_vector(v.value, 1 + len(obs), "affine coefficient candidate")
                for n, (_, fixed, obs, v) in self.rules.items()
                if fixed is None
            }
            if any(value is None for value in values.values()):
                return None, report
            report["numerical_minimization_cost"] = finite(
                problem.value, "solver objective"
            )
            report["normalized_coefficients"] = values
            return values, report
        except (cp.error.SolverError, ValueError, OverflowError) as exc:
            report.update(status="unresolved", message=str(exc))
            return None, report

    def controller(self, values, name, simplify):
        rules = []
        for n, (control, fixed, observes, _) in self.rules.items():
            if fixed is not None:
                rules.append(AffineControlRule(n, control.unit, fixed))
                continue
            coefficients = [F(v) for v in values[n]]
            if simplify:
                coefficients = [
                    v.limit_denominator(1_000_000)
                    if abs(v - v.limit_denominator(1_000_000))
                    <= F(self.model.solver_tolerance)
                    else v
                    for v in coefficients
                ]
            span = F(control.upper) - F(control.lower)
            terms = tuple(
                AffineTerm(
                    k,
                    finite(
                        float(span * a / (self.ranges[k][1] - self.ranges[k][0])),
                        "controller coefficient",
                    ),
                    f"{control.unit}/{self.model.input_space.coordinate(k).unit}",
                )
                for k, a in zip(observes, coefficients[1:])
                if a
            )
            offset = (
                F(control.lower)
                + span * coefficients[0]
                - sum(
                    (F(t.coefficient) * self.ranges[t.variable][0] for t in terms), F(0)
                )
            )
            rules.append(
                AffineControlRule(
                    n, control.unit, finite(float(offset), "controller offset"), terms
                )
            )
        return AffineController(
            name,
            tuple(rules),
            "CVXPY affine-policy synthesis; physical float coefficients independently audited; no certified policy optimum",
        )


def _objective_bound(claim, objective, scenarios):
    model, controller = claim.adapter.model, claim.adapter.controller
    continuous = type(claim.domain) is not FiniteSet
    envelope = ControllerEnvelope(claim) if continuous else None
    sign = (
        -1
        if type(objective) is LinearObjective and objective.sense == "maximize"
        else 1
    )
    values, enclosures = [], []
    errors = (
        {n: triple[2] for n, triple in envelope.commands.items()} if continuous else {}
    )
    for scenario in scenarios:
        point = scenario.values
        commands = controller.commands(point)
        if type(objective) is LinearObjective:
            value = sign * model.output(objective.output)._exact_value(
                {**point, **commands}
            )
        else:
            value = sum(
                (
                    F(t.weight)
                    * ((F(commands[t.control]) - F(t.target)) / F(t.scale)) ** 2
                    for t in objective.targets
                ),
                F(0),
            )
            if continuous:
                ideal = {
                    r.control: F(r.offset)
                    + exact_dot(
                        (t.coefficient for t in r.terms),
                        (point[t.variable] for t in r.terms),
                    )
                    for r in controller.rules
                }
                enclosures.append(
                    sum(
                        (
                            F(t.weight)
                            * (
                                (
                                    abs(ideal[t.control] - F(t.target))
                                    + errors[t.control]
                                )
                                / F(t.scale)
                            )
                            ** 2
                            for t in objective.targets
                        ),
                        F(0),
                    )
                )
        values.append(value)
    lower = max(values)
    if not continuous:
        upper = lower
    elif type(objective) is LinearObjective:
        offset, coefficients, error = envelope.output_form(
            model.output(objective.output)
        )
        upper = (
            envelope.upper(
                sign * offset, {n: sign * a for n, a in coefficients.items()}
            )
            + error
        )
    else:
        upper = max(enclosures)
    if lower > upper:
        raise ValueError("objective enclosure contradicts rounded generator execution")
    low, high = (lower, upper) if sign == 1 else (-upper, -lower)
    return Evidence(
        "objective",
        "controller_objective",
        "rounded_controller_worst_case_enclosure",
        "verified",
        details={
            "declaration": objective.to_dict(),
            "aggregation": "minimum" if sign == -1 else "maximum",
            "unit": model.output(objective.output).unit
            if type(objective) is LinearObjective
            else "1",
            "lower": round_down(low),
            "upper": round_up(high),
            "lower_exact": str(low),
            "upper_exact": str(high),
            "guaranteed_value": round_down(low) if sign == -1 else round_up(high),
            "scope": "worst-case performance of this saved rounded controller over the declared domain; not bounds on the optimum over controllers",
            "optimality": "not_certified",
            "generator_minimization_values_exact": list(map(str, values)),
            "command_rounding_errors_exact": {n: str(e) for n, e in errors.items()},
            "tracking_generator_enclosures_exact": list(map(str, enclosures)),
            "support_bounds": [] if envelope is None else envelope.proofs,
            "derivation": "Finite replay is exact. Continuous affine output uses its signed affine enclosure plus command rounding; tracking uses the maximum over generators of sum(weight*((abs(ideal_command-target)+rounding_error)/scale)**2), a convex upper enclosure throughout their convex hull.",
        },
    )


def synthesize(claim, *, objective, name, max_vertices):
    request = {
        "query": "synthesize_controller",
        "objective": objective.to_dict()
        if type(objective) in (LinearObjective, ControlTrackingObjective)
        else objective,
        "name": name,
        "max_vertices": max_vertices,
        "backend": "cvxpy",
    }
    evidence, attempts, controller, audit = [], [], None, None

    def result(execution, message=None):
        return ControllerSynthesis(
            claim,
            request,
            execution,
            controller,
            audit,
            tuple(evidence),
            ()
            if message is None
            else (
                Diagnostic("controller_synthesis_" + execution, "controller", message),
            ),
        )

    try:
        model, objective, scenarios = _validate(claim, objective, name, max_vertices)
        request["objective"] = objective.to_dict()
        cp = load_cvxpy()
    except (NotImplementedError, BackendUnavailable) as exc:
        return result("unsupported", str(exc))
    except (ValueError, TypeError, OverflowError) as exc:
        return result("invalid", str(exc))
    evidence.append(
        Evidence(
            "search",
            "controller_synthesis",
            "complete_generator_affine_policy_search",
            "verified",
            details={
                "generator_count": len(scenarios),
                "generators": [s.to_dict() for s in scenarios],
                "policy_class": "one static affine physical command per control; only permitted observations; fixed commands retained",
                "scope": "proposal over every declared generator; acceptance requires a separate whole-domain audit of the exported float coefficients and rounded commands",
            },
        )
    )
    try:
        program = PolicyProgram(cp, claim, model, objective, scenarios)
        for reserve in (0.5, 0.0):
            values, report = program.solve(claim.requirements, reserve)
            attempts.append(report)
            if values is None:
                continue
            for simplify in (True, False):
                controller = program.controller(values, name, simplify)
                audit = None
                candidate = claim.with_controller(controller)
                audit = candidate.audit_result()
                attempts.append(
                    {
                        "controller_ref": controller.ref.to_dict(),
                        "simplified_coefficients": simplify,
                        "audit_ref": audit.ref.to_dict(),
                        "verdict": audit.payload.verdict,
                        "execution": audit.execution,
                    }
                )
                if audit.execution == "completed" and audit.payload.verdict == "pass":
                    evidence.append(_objective_bound(candidate, objective, scenarios))
                    evidence.append(
                        Evidence(
                            "solver",
                            "controller_synthesis",
                            "cvxpy_candidate_search",
                            "unresolved",
                            details={"attempts": attempts},
                        )
                    )
                    return result("completed")
    except (ValueError, TypeError, OverflowError, RuntimeError) as exc:
        attempts.append({"status": "unresolved", "message": str(exc)})
    evidence.append(
        Evidence(
            "solver",
            "controller_synthesis",
            "cvxpy_candidate_search",
            "unresolved",
            details={"attempts": attempts},
        )
    )
    return result(
        "unresolved",
        "No controller with a passing rounded-command audit and objective enclosure was found. This does not prove affine-policy or adjustable-operation infeasibility, even if the numerical solver reported infeasible.",
    )
