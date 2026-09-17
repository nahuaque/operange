"""Prepare, generate and check operating candidates before making result records.

The solver model is disposable. Checked points own a separate source clone,
which downstream numerical analyses can use without decoding exported results.
Calibration may reuse the solver and parameter mechanics while retaining its
measurement-specific acceptance rules.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..contract_types import Diagnostic, freeze, unique
from ..primitives import finite
from ._idaes_model import backend, component, number, uid, unit
from ._idaes_workflow import InitializationFailure, diagnostics, initialize


class SolverUnavailable(RuntimeError):
    pass


def solve(model, settings):
    pyo, _ = backend()
    from pyomo.common.modeling import unique_component_name

    if (
        next(
            model.component_data_objects(pyo.Objective, active=True, descend_into=True),
            None,
        )
        is None
    ):
        model.add_component(
            unique_component_name(model, "_operange_feasibility_objective"),
            pyo.Objective(expr=0),
        )
    solver = pyo.SolverFactory(settings.solver)
    if not solver.available(exception_flag=False):
        raise SolverUnavailable(
            f"Solver {settings.solver!r} is unavailable; install/configure it separately"
        )
    result = solver.solve(
        model, options=dict(settings.solver_options), load_solutions=False
    )
    termination = str(result.solver.termination_condition)
    details = {
        "solver": settings.solver,
        "status": str(result.solver.status),
        "termination": termination,
        "scope": "candidate generation only",
    }
    if termination not in {"optimal", "locallyOptimal", "globallyOptimal", "feasible"}:
        return False, details
    if not len(result.solution):
        return False, {**details, "reason": "No fresh solver solution was returned"}
    model.solutions.load_from(result)
    return True, details


@dataclass(frozen=True)
class OperatingPointRequest:
    realization: Mapping[str, float]
    commands: Mapping[str, float | None]
    requirements: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "realization", freeze(self.realization))
        object.__setattr__(self, "commands", freeze(self.commands))
        object.__setattr__(self, "requirements", tuple(self.requirements))
        unique(self.requirements, "selected requirements")
        for value in self.commands.values():
            if value is not None:
                finite(value, "actuator command")

    @classmethod
    def from_claim(cls, claim, point):
        return cls(
            point,
            {
                b.name: claim.recourse.rule(b.name).fixed_value
                for b in claim.adapter.controls
            },
            claim.requirements,
        )


@dataclass(frozen=True)
class Candidate:
    model: object
    solved: bool
    details: Mapping


@dataclass(frozen=True)
class CheckedPoint:
    model: object
    values: Mapping[str, float]
    residuals: tuple
    objective: Mapping | None = None

    @property
    def feasible(self):
        return all(
            (abs(value) if row.side == "eq" else value) <= row.tolerance
            for row, value in self.residuals
        )


@dataclass
class PointExecution:
    """Internal lifecycle; public feasibility and evidence are assembled later."""

    adapter: object
    request: OperatingPointRequest
    rows: tuple = field(init=False)
    model: object = None
    source: object = None
    checked: CheckedPoint | None = None
    candidate: Candidate | None = None
    before: list = field(default_factory=list)
    after: list = field(default_factory=list)
    stage: str = "scenario_setup"
    iterate_origin: str = "source_start"
    failure: Diagnostic | None = None

    def __post_init__(self):
        self.rows = tuple(
            r
            for r in self.adapter._rows
            if r.kind != "requirement" or r.name in self.request.requirements
        )

    def prepare(self, *, initialize_model=True):
        pyo, _ = backend()
        adapter, request = self.adapter, self.request
        # Requests are internal but still reject incomplete command vectors.
        adapter.input_space.validate(request.realization)
        if set(request.requirements) - {r.name for r in adapter.requirements}:
            raise ValueError("Operating point references an undeclared requirement")
        if set(request.commands) != {b.name for b in adapter.controls}:
            raise ValueError(
                "Operating point needs every actuator command or permission"
            )
        self.model = model = adapter._template.clone()
        for binding in adapter.inputs:
            item = component(model, binding.component)
            if item.ctype is pyo.Var:
                item.fix(request.realization[binding.name])
            else:
                item.set_value(request.realization[binding.name])
        for binding in adapter.controls:
            item = component(model, binding.component)
            command = request.commands[binding.name]
            item.unfix() if command is None else item.fix(command)
        for req in adapter.requirements:
            if req.name not in request.requirements:
                component(model, req.component).deactivate()
        diagnostics(
            model,
            adapter,
            self.before,
            stage="before_initialization",
            rows=self.rows,
            iterate_origin=self.iterate_origin,
        )
        self.stage = "initialization"
        if initialize_model:
            initialize(model, adapter, self.before)
        self.source = model.clone()
        self.iterate_origin = "prepared_start_without_solver_solution"
        self.stage = "solve"
        return model

    def generate_candidate(self):
        solved, details = solve(self.model, self.adapter)
        return Candidate(self.model, solved, details)

    def check_candidate(self, candidate, *, objective=False):
        self.stage = "candidate_check"
        if self.source is None or not candidate.solved:
            raise ValueError(
                "Candidate checking requires a prepared source and a returned solver candidate"
            )
        pyo, _ = backend()
        checker = self.source
        candidates = []
        for var in checker.component_data_objects(
            pyo.Var, active=True, descend_into=True
        ):
            value = number(component(candidate.model, uid(var)))
            if var.fixed and value != number(var):
                raise ValueError(
                    f"Solver changed a fixed input, command or state: {uid(var)}"
                )
            candidates.append((var, value))
        for var, value in candidates:
            var.set_value(value, skip_validation=True)
        self.iterate_origin = "returned_solver_candidate"
        values = dict(self.request.realization)
        values.update(
            (b.name, number(component(checker, b.component)))
            for b in self.adapter.controls + self.adapter.outputs
        )
        values.update(
            (name, number(component(checker, path)))
            for name, path, _ in self.adapter._states
        )
        residuals = tuple((row, number(row.expression(checker))) for row in self.rows)
        self.checked = CheckedPoint(checker, values, residuals)
        if self.checked.feasible and objective:
            obj = next(
                checker.component_data_objects(
                    pyo.Objective, active=True, descend_into=True
                )
            )
            self.checked = CheckedPoint(
                checker,
                values,
                residuals,
                {
                    "component": uid(obj),
                    "sense": str(obj.sense),
                    "unit": unit(obj.expr),
                    "value": number(obj.expr),
                    "scope": "Value at the checked shared-command candidate; no optimality certificate.",
                },
            )
        return self.checked

    @property
    def feasible(self):
        return (
            self.failure is None and self.checked is not None and self.checked.feasible
        )

    def require_checked(self):
        if not self.feasible:
            raise ValueError("Operating point has no checked feasible source solution")
        return self.checked

    def run(self, *, candidate=None, objective=False):
        try:
            self.prepare(initialize_model=candidate is None)
            self.candidate = (
                self.generate_candidate() if candidate is None else candidate
            )
            if self.candidate.solved:
                self.check_candidate(self.candidate, objective=objective)
                if not self.checked.feasible:
                    self.failure = Diagnostic(
                        "idaes_residual_violation",
                        self.stage,
                        "Returned candidate violates a source constraint or bound; no infeasibility conclusion follows.",
                    )
            else:
                self.failure = Diagnostic(
                    "idaes_solver_failed",
                    self.stage,
                    f"Solver termination {self.candidate.details['termination']}; no feasible candidate or infeasibility proof established.",
                )
        except Exception as exc:
            code = "idaes_evaluation_error"
            if isinstance(exc, InitializationFailure):
                code = exc.code
            elif isinstance(exc, SolverUnavailable):
                code = "idaes_solver_unavailable"
            else:
                code = {
                    "initialization": "idaes_initialization_failed",
                    "solve": "idaes_solver_error",
                    "candidate_check": "idaes_candidate_error",
                }.get(self.stage, code)
            self.failure = Diagnostic(code, self.stage, f"{type(exc).__name__}: {exc}")
        if self.model is not None:
            diagnostics(
                self.source if self.source is not None else self.model,
                self.adapter,
                self.after,
                stage="checked_candidate"
                if self.feasible
                else f"unresolved_{self.stage}",
                rows=self.rows,
                iterate_origin=self.iterate_origin,
            )
        return self


def execute_point(adapter, request, *, candidate=None, objective=False):
    return PointExecution(adapter, request).run(
        candidate=candidate, objective=objective
    )
