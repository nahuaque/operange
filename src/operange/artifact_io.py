"""Strict, declarative artifact loading across supported versioned schemas.

Dispatch chooses a built-in reader only. The artifact's type owns its semantic
validation; loading never runs a model, solver, calibration, or certificate check.
Reports that only support export are not made replayable by this entry point.
"""

from importlib import import_module

from ._artifact_json import read_json


# Explicit allowlist, not an extensible object deserializer. Module and attribute
# names are source code; documents cannot supply Python classes or import paths.
_READERS = {
    "process_result/v1": ("engineering_results", "result_from_dict"),
    "process_result_bundle/v1": ("engineering_results", "result_from_dict"),
    "affine_controller/v1": ("controllers", "AffineController.from_dict"),
    "storage_controller/v1": ("storage_controllers", "StorageController.from_dict"),
    "frozen_controller/v1": ("frozen_controller", "FrozenController.from_dict"),
    "process_network/v1": ("network", "ProcessNetwork.from_dict"),
    "engineering_change_comparison/v1": ("changes", "ChangeComparison.from_dict"),
    "operating_limit/v1": ("operating_limits", "OperatingLimitResult.from_dict"),
    "dispatch_sensitivity/v1": (
        "dispatch_sensitivity",
        "DispatchSensitivityResult.from_dict",
    ),
    "idaes_linear_export/v1": (
        "integrations._idaes_affine",
        "IdaesLinearExport.from_dict",
    ),
    "idaes_shared_controls/v1": (
        "integrations._idaes_shared_result",
        "IdaesSharedControlResult.from_dict",
    ),
    "idaes_local_uncertainty/v1": (
        "integrations._idaes_uncertainty_result",
        "IdaesUncertaintyResult.from_dict",
    ),
    "calibration_result/v1": ("integrations.parmest", "CalibrationResult.from_dict"),
    "calibration_bootstrap/v1": (
        "integrations.parmest",
        "CalibrationBootstrap.from_dict",
    ),
    "calibration_diagnostics/v1": (
        "integrations._calibration_quality",
        "CalibrationDiagnostics.from_dict",
    ),
    "calibration_validation/v1": (
        "integrations._calibration_quality",
        "CalibrationValidation.from_dict",
    ),
}


def artifact_from_json(document):
    """Load a supported artifact using its schema, without executing its contents.

    Content identities establish consistency, not authorship or mathematical
    correctness. Use ``verify_result`` separately for supported certificates.
    Optional numerical backends are not required to load their exported records.
    """
    return _artifact_from_dict(read_json(document))


def _artifact_from_dict(data, *, schemas=None):
    if not isinstance(data, dict):
        raise ValueError("artifact must be a JSON object")
    schema = data.get("schema_version")
    if type(schema) is int and schema == 1 and schemas is None:
        # The original heat-recovery format uses an integer schema plus model_id.
        from .policy import FrozenHeatRecoveryContract

        return FrozenHeatRecoveryContract.from_dict(data)
    if (
        not isinstance(schema, str)
        or schema not in _READERS
        or (schemas is not None and schema not in schemas)
    ):
        raise ValueError("unsupported artifact schema")
    module, attribute = _READERS[schema]
    reader = import_module(f".{module}", "operange")
    for name in attribute.split("."):
        reader = getattr(reader, name)
    return reader(data)
