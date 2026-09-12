"""Conservative comparison of audited engineering declarations.

Built-in manifests have known locations for service, equipment and model data.
Unknown adapters still get complete structural differences, but changing their
declarations cannot establish preservation of service by guessing from names.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .contract_types import Record, freeze, plain


@dataclass(frozen=True)
class FieldChange(Record):
    """One structural change at a JSON Pointer in the audited contract or metric."""

    path: str
    before_present: bool
    after_present: bool
    before: object
    after: object


@dataclass(frozen=True)
class ContractComparison(Record):
    changed_sections: tuple[str, ...]
    domain_preserved: bool
    requirements_preserved: bool | None
    model_preserved: bool | None
    commitment: Literal["preserved", "revised", "unknown"]
    differences: tuple[FieldChange, ...]
    notes: tuple[str, ...]


def _differences(before, after, path=""):
    if before == after:
        return ()
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        changes = []
        for key in sorted(set(before) | set(after)):
            pointer = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in before or key not in after:
                changes.append(
                    FieldChange(
                        pointer,
                        key in before,
                        key in after,
                        before.get(key),
                        after.get(key),
                    )
                )
            else:
                changes.extend(_differences(before[key], after[key], pointer))
        return tuple(changes)
    # Ordered sequences remain whole values; positions are not stable identities.
    return (FieldChange(path, True, True, before, after),)


def _selected(contract):
    refs = contract.operating.get("requirement_refs")
    if refs is None:
        refs = tuple(
            c.constraint_id for c in contract.constraints if c.kind == "requirement"
        )
    return {
        c.constraint_id: {
            "constraint": c.to_dict(),
            "quantity": contract.quantity(c.residual_ref).to_dict(),
        }
        for c in contract.constraints
        if c.kind == "requirement" and c.constraint_id in refs
    }


def _by_name(records, key="name"):
    return {item[key]: item for item in records}


def _basis(contract):
    """Partition known declarations without changing their source contracts."""
    model, design = plain(contract.model), plain(contract.design)
    operating = plain(contract.operating)
    selected = _selected(contract)
    service = {"selected_requirements": selected}
    metadata = {"model_id": contract.model_id}
    known = True
    family = contract.model_id
    try:
        if family in ("affine_process/v1", "linear_process/v1"):
            declared = _by_name(model.pop("requirements"))
            service["definitions"] = {name: declared[name] for name in selected}
            metadata["name"] = model.pop("name")
            # Input-space provenance/scales are metadata; the audited domain owns
            # its actual input support. Preserve units/axes in the model signature.
            coordinates = model.pop("input_space")["coordinates"]
            metadata["input_coordinates"] = {
                c["name"]: {
                    k: v
                    for k, v in c.items()
                    if k in ("provenance", "nominal", "scale")
                }
                for c in coordinates
            }
            model["input_coordinates"] = {
                c["name"]: {
                    k: v
                    for k, v in c.items()
                    if k not in ("provenance", "nominal", "scale")
                }
                for c in coordinates
            }
            if family == "linear_process/v1":
                controls = model.pop("controls")
                design["control_bounds"] = {
                    c["name"]: {"lower": c["lower"], "upper": c["upper"]}
                    for c in controls
                }
                model["control_definitions"] = {
                    c["name"]: {
                        k: v for k, v in c.items() if k not in ("lower", "upper")
                    }
                    for c in controls
                }
                design["operating_limits"] = _by_name(model.pop("operating_limits"))
                model.pop("solver_tolerance")  # Compared as numerical policy below.
                model.pop("objective", None)  # An operating preference, not physics.
            else:
                model["controls"] = _by_name(model["controls"], "quantity_id")
            model["outputs"] = _by_name(model["outputs"])
        elif family in (
            "heat_recovery_constant_cop_v1",
            "thermal_storage_two_period_v1",
        ):
            service["definition"] = model.pop("requirement")
            if family == "thermal_storage_two_period_v1":
                service["durations"] = {
                    name: model.pop(name)
                    for name in ("preparation_hours", "event_hours")
                }
        elif model.get("kind") == "supplied_startup_profiles":
            family = "supplied_startup_profiles"
            service["events"] = _by_name(model.pop("events"))
            service["horizon_seconds"] = operating["horizon_seconds"]
            service["background_load"] = design.pop("background_load")
        elif model.get("kind") == "sensible_heat_cascade":
            family = "sensible_heat_cascade"
            service["streams"] = _by_name(model.pop("streams"))
        else:
            known = False
    except (KeyError, TypeError):
        # A consumer may reuse a model ID with a different schema. Its fields
        # remain reviewable, but the built-in interpretation does not apply.
        return None
    if not known:
        return None
    # Equations are part of the physical interpretation even if a consumer
    # changes only the registry while retaining a built-in model identifier.
    model["equation_declarations"] = {
        c.constraint_id: c.to_dict()
        for c in contract.constraints
        if c.kind == "equation"
    }
    return freeze(
        {
            "family": family,
            "model": model,
            "design": design,
            "requirements": service,
            "operating": operating,
            "domain": contract.domain,
            "numerical": contract.numerical_policy,
            "metadata": metadata,
        }
    )


def compare_contracts(before, after, before_distance=None, after_distance=None):
    old = {"contract": before.to_dict(), "distance": plain(before_distance)}
    new = {"contract": after.to_dict(), "distance": plain(after_distance)}
    differences = _differences(old, new)
    domain_same = before.domain == after.domain
    selected_same = _selected(before) == _selected(after)
    first, second = _basis(before), _basis(after)
    sections, notes = [], []
    if first is not None and second is not None and first["family"] == second["family"]:
        for section in (
            "model",
            "design",
            "requirements",
            "operating",
            "domain",
            "numerical",
            "metadata",
        ):
            if first[section] != second[section]:
                sections.append(section)
        requirements_same = first["requirements"] == second["requirements"]
        model_same = first["model"] == second["model"]
        # For known manifests the generated registry is fixed by declarations.
        # Still retain unexpected registry edits as unknown model changes.
        old_quantities = {
            q.quantity_id: (q.unit, q.physical_kind, q.role) for q in before.quantities
        }
        new_quantities = {
            q.quantity_id: (q.unit, q.physical_kind, q.role) for q in after.quantities
        }
        common = set(old_quantities) & set(new_quantities)
        if any(old_quantities[q] != new_quantities[q] for q in common):
            model_same = False
            if "model" not in sections:
                sections.append("model")
    else:
        for section, a, b in (
            (
                "model",
                (before.model_id, before.model, before.quantities, before.constraints),
                (after.model_id, after.model, after.quantities, after.constraints),
            ),
            ("design", before.design, after.design),
            ("requirements", _selected(before), _selected(after)),
            ("operating", before.operating, after.operating),
            ("domain", before.domain, after.domain),
            ("numerical", before.numerical_policy, after.numerical_policy),
        ):
            if a != b:
                sections.append(section)
        identical = before == after
        requirements_same = True if identical else False if not selected_same else None
        model_same = True if identical else None
        if not identical:
            notes.append(
                "Changed unknown or different adapter families have no established service-equivalence interpretation."
            )
    if before_distance != after_distance:
        sections.append("distance")
    if not domain_same or requirements_same is False:
        commitment = "revised"
    elif requirements_same is True and model_same is True:
        commitment = "preserved"
    else:
        commitment = "unknown"
    if not domain_same:
        notes.append(
            "The declared domain changed; no subset, equivalence or probability assumption is inferred."
        )
    if requirements_same is False:
        notes.append(
            "Selected service declarations changed, including their thresholds, tolerances or temporal scope."
        )
    if model_same is False:
        notes.append(
            "Physical model declarations changed; preservation of the original commitment is not established."
        )
    if "operating" in sections:
        notes.append(
            "The candidate uses its own declared operating permissions; the original policy is not claimed to work."
        )
    if "distance" in sections:
        notes.append(
            "Distance settings changed; this comparison runs audits, not distance searches."
        )
    return ContractComparison(
        tuple(sections),
        domain_same,
        requirements_same,
        model_same,
        commitment,
        differences,
        tuple(notes),
    )
