"""Scenario replication and equipment relief on the common normalized system."""

from dataclasses import dataclass, replace
from fractions import Fraction as F

from ._linear_problem import LinearSystem, Variable


@dataclass(frozen=True)
class ScenarioBlock:
    offset: int
    mapping: tuple[tuple[str, str], ...]


def stack_scenarios(systems):
    """Replicate unknowns while retaining each row's physical/scenario origin."""
    systems = tuple(systems)
    if any(len(system.sources) != len(system.rows) for system in systems):
        raise ValueError("scenario replication requires physical row sources")
    variables, rows, upper, sources, blocks = [], [], [], [], []
    width = sum(len(system.variables) for system in systems)
    for i, system in enumerate(systems):
        offset = len(variables)
        renamed = tuple(
            replace(c, name=f"_operange_dispatch_{i}_{j}", scenario=i)
            for j, c in enumerate(system.variables)
        )
        blocks.append(
            ScenarioBlock(
                offset,
                tuple((c.name, new.name) for c, new in zip(system.variables, renamed)),
            )
        )
        variables.extend(renamed)
        for row, bound, source in zip(system.rows, system.upper, system.sources):
            rows.append(
                (F(0),) * offset + tuple(row) + (F(0),) * (width - offset - len(row))
            )
            upper.append(bound)
            sources.append(replace(source, scenario=i))
    return LinearSystem(
        tuple(variables), {}, rows, upper, [s.legacy_ref for s in sources], sources
    ), tuple(blocks)


@dataclass(frozen=True)
class ReliefColumn:
    variable: Variable
    constraint: str


def augment_relief(system, columns):
    """One bounded auxiliary per physical relaxation, shared across matching rows."""
    columns = tuple(columns)
    if len(system.sources) != len(system.rows):
        raise ValueError("relief requires physical row sources")
    if len({c.constraint for c in columns}) != len(columns):
        raise ValueError("duplicate relief target")
    for column in columns:
        variable = column.variable
        targets = [
            s
            for s in system.sources
            if s.kind == "operating_limit" and s.name == column.constraint
        ]
        if (
            not targets
            or variable.role != "auxiliary"
            or variable.lower != 0
            or variable.upper <= 0
            or any(s.unit != variable.unit for s in targets)
        ):
            raise ValueError(
                "relief must target a physical operating limit in native units"
            )
    rows = [
        row
        + tuple(
            -F(c.variable.upper) / source.scale
            if source.kind == "operating_limit" and source.name == c.constraint
            else F(0)
            for c in columns
        )
        for row, source in zip(system.rows, system.sources)
    ]
    return replace(
        system,
        variables=system.variables + tuple(c.variable for c in columns),
        rows=rows,
    )


def relax_rows(system, amounts):
    """Apply physical relief amounts to already normalized operating rows only."""
    if len(system.sources) != len(system.rows):
        raise ValueError("relief requires physical row sources")
    if set(amounts) - {s.name for s in system.sources if s.kind == "operating_limit"}:
        raise ValueError("relief must name physical operating limits")
    if any(F(value) < 0 for value in amounts.values()):
        raise ValueError("negative relief")
    return replace(
        system,
        upper=tuple(
            b + F(amounts[source.name]) / source.scale
            if source.kind == "operating_limit" and source.name in amounts
            else b
            for b, source in zip(system.upper, system.sources)
        ),
    )
