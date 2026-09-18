"""Transport is shared; loading, engineering semantics and proof remain distinct."""

import ast
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess  # nosec B404
import sys

import pytest

from operange import (
    AffineControlRule,
    AffineController,
    ChangeComparison,
    DispatchChange,
    DispatchSensitivityResult,
    EvaluationResult,
    OperatingLimitResult,
    ProcessNetwork,
    RobustnessResult,
    SensitivityResult,
    artifact_from_json,
    domain_from_json,
    result_from_json,
    verify_result,
)
from operange.contract_types import digest
from operange.integrations.idaes import (
    IdaesLinearExport,
    IdaesSharedControlResult,
    IdaesUncertaintyResult,
)
from operange.integrations.parmest import (
    CalibrationBootstrap,
    CalibrationDiagnostics,
    CalibrationResult,
    CalibrationValidation,
)
from operange.reference import FrozenHeatRecoveryContract


_FLAT = {
    "calibration_result": CalibrationResult,
    "calibration_bootstrap": CalibrationBootstrap,
    "calibration_diagnostics": CalibrationDiagnostics,
    "calibration_validation": CalibrationValidation,
    "idaes_shared_controls": IdaesSharedControlResult,
    "idaes_local_uncertainty": IdaesUncertaintyResult,
}


@pytest.fixture(scope="module")
def corpus():
    from examples.operating_roles import example
    from examples.storage_replay import preparation
    from examples.dispatch_sensitivity import example as dispatch
    from tests.test_operating_limits import simple

    directory = Path(__file__).with_name("fixtures")
    documents = {
        name: (cls, (directory / f"{name}_v1.json").read_text())
        for name, cls in _FLAT.items()
    }
    legacy = json.loads((directory / "prototype_result_v1.json").read_text())
    for name, cls in (
        ("evaluation", EvaluationResult),
        ("sensitivity", SensitivityResult),
        ("robustness", RobustnessResult),
        ("frozen_heat", FrozenHeatRecoveryContract),
    ):
        documents[name] = cls, json.dumps(legacy[name])
    network, _, cases = example()
    model = network.compile()
    controller = AffineController(
        "Fixed fuel", (AffineControlRule("fuel", "MW", 8),), "Transport test"
    )
    claim = model.as_claim(cases, controller=controller)
    dispatch_model, domain = dispatch()
    results = {
        "controller": controller,
        "storage_controller": preparation(2),
        "network": network,
        "frozen": claim.freeze(),
        "linear_export": IdaesLinearExport(model, model.base_contract, ("fuel",)),
        "changes": ChangeComparison(result_from_json(documents["robustness"][1]), ()),
        "limit": simple().operating_limit("load", max_iterations=3),
        "dispatch_sensitivity": dispatch_model.as_claim(domain).dispatch_sensitivity(
            {"load": 16}, changes=(DispatchChange("input", "load", "MW", 2),)
        ),
    }
    documents.update(
        {name: (type(value), value.to_json()) for name, value in results.items()}
    )
    return documents


_NAMES = (
    *_FLAT,
    "evaluation",
    "sensitivity",
    "robustness",
    "frozen_heat",
    "controller",
    "storage_controller",
    "network",
    "frozen",
    "linear_export",
    "changes",
    "limit",
    "dispatch_sensitivity",
)


@pytest.mark.parametrize("name", _NAMES)
def test_typed_and_schema_readers_preserve_artifacts_without_execution(
    corpus, name, monkeypatch
):
    from operange import Claim

    def forbidden(*args, **kwargs):
        raise AssertionError("artifact loading must not execute a study")

    for method in ("evaluate_result", "audit_result", "synthesize_controller"):
        monkeypatch.setattr(Claim, method, forbidden)
    monkeypatch.setattr(ProcessNetwork, "compile", forbidden)
    monkeypatch.setattr(AffineController, "commands", forbidden)
    cls, document = corpus[name]
    original = cls.from_json(document)
    restored = artifact_from_json(document)
    assert type(restored) is cls  # nosec B101
    assert restored.to_json() == original.to_json()  # nosec B101
    data = json.loads(document)
    before = deepcopy(data)
    assert cls.from_dict(data).to_json() == original.to_json()  # nosec B101
    assert data == before  # nosec B101
    if hasattr(original, "ref"):
        assert restored.ref == original.ref  # nosec B101
    if cls in (
        EvaluationResult,
        SensitivityResult,
        RobustnessResult,
        ChangeComparison,
        OperatingLimitResult,
        DispatchSensitivityResult,
    ):
        compact = original.to_json(compact=True)
        assert artifact_from_json(compact).to_json() == original.to_json()  # nosec B101


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize(
    "bad",
    [
        "duplicate",
        "nested_duplicate",
        "NaN",
        "Infinity",
        "-Infinity",
        "1e999",
        "-1e999",
    ],
)
def test_all_readers_reject_ambiguous_or_nonfinite_json(corpus, name, bad):
    cls, document = corpus[name]
    if bad == "duplicate":
        schema = json.dumps(json.loads(document)["schema_version"])
        document = '{"schema_version":' + schema + "," + document.lstrip()[1:]
    else:
        value = '{"x": 0, "x": 0}' if bad == "nested_duplicate" else bad
        document = '{"unused":' + value + "," + document.lstrip()[1:]
    for reader in (cls.from_json, artifact_from_json):
        with pytest.raises(ValueError, match="duplicate|nonfinite"):
            reader(document)


@pytest.mark.parametrize("name", _NAMES)
def test_readers_reject_wrong_schema_fields_and_identity(corpus, name):
    cls, document = corpus[name]
    data = json.loads(document)
    corruptions = [dict(data, schema_version="unknown/v99"), dict(data, unexpected=0)]
    for key in (
        "result_id",
        "controller_id",
        "network_id",
        "frozen_id",
        "comparison_id",
        "fingerprint",
    ):
        if key in data:
            corruptions.extend(
                (
                    dict(data, **{key: "sha256:" + "0" * 64}),
                    {k: v for k, v in data.items() if k != key},
                )
            )
    for corrupted in corruptions:
        for reader in (cls.from_json, artifact_from_json):
            with pytest.raises(ValueError):
                reader(json.dumps(corrupted))
    for wrong_shape in ("[]", "null", '"artifact"', "1"):
        with pytest.raises(ValueError):
            cls.from_json(wrong_shape)


@pytest.mark.parametrize("name", _FLAT)
def test_fresh_identity_does_not_override_family_semantic_validation(corpus, name):
    cls, document = corpus[name]
    data = json.loads(document)
    if name == "calibration_result":
        data["objective_value"] = -1
    else:
        data["scope"]["confidence_level"] = 0.95
    body = {k: v for k, v in data.items() if k not in ("schema_version", "result_id")}
    data["result_id"] = "sha256:" + digest(body)
    for reader in (cls.from_json, artifact_from_json):
        with pytest.raises(ValueError, match="scope|qualifications|objective"):
            reader(json.dumps(data))


def test_loading_is_not_certificate_verification(corpus):
    result = artifact_from_json(corpus["dispatch_sensitivity"][1]).base
    evidence = tuple(
        replace(e, method="unproved") if e.evidence_id == "operating_objective" else e
        for e in result.evidence
    )
    forged = replace(result, evidence=evidence)
    assert forged.ref != result.ref  # nosec B101
    restored = artifact_from_json(forged.to_json())
    assert restored.ref == forged.ref  # nosec B101
    assert not verify_result(restored).verified  # nosec B101
    # A generic reader is not permission to treat a calibration as a proof.
    assert verify_result(corpus["calibration_result"][1]).status == "invalid"  # nosec B101


def test_domain_reader_uses_the_same_strict_numbers():
    for document in ('{"kind":"box","kind":"box"}', '{"unused":1e999}'):
        with pytest.raises(ValueError, match="duplicate|nonfinite"):
            domain_from_json(document)


def test_generic_reader_needs_no_optional_backends(corpus):
    script = """
import importlib.abc, json, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'idaes', 'pyomo', 'pandas', 'highspy', 'cvxpy'}:
            raise ImportError('optional backend deliberately unavailable')
sys.meta_path.insert(0, BlockOptional())
from operange import artifact_from_json
for document in json.loads(sys.stdin.read()):
    restored = artifact_from_json(document)
    if artifact_from_json(restored.to_json()).to_json() != restored.to_json():
        raise RuntimeError('artifact changed')
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        input=json.dumps([document for _, document in corpus.values()]),
        text=True,
        capture_output=True,
        check=True,
    )  # nosec B603


def test_transport_has_no_semantic_dependencies_or_controller_helper_imports():
    root = Path(__file__).parents[1] / "src/operange"
    transport = ast.parse((root / "_artifact_json.py").read_text())
    assert not any(
        isinstance(n, ast.ImportFrom) and n.level for n in ast.walk(transport)
    )  # nosec B101
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == "controllers":
                assert not {n.name for n in node.names} & {"_fields", "_read_json"}  # nosec B101
