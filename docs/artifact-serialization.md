# Artifact serialization and result meaning

`artifact_from_json` selects a supported reader from the document's schema:

```python
from operange import EvaluationResult, artifact_from_json, verify_result

saved = claim.evaluate_result(realization).to_json()
result = artifact_from_json(saved)
if isinstance(result, EvaluationResult):
    print(result.execution, result.payload.feasibility)
    report = verify_result(result)
```

The typed `from_json` methods remain available when a consumer expects a
particular family. `result_from_json` still accepts only evaluation, sensitivity,
and robustness results, including their compact bundles. The generic reader
returns the original type; it does not assign a common success or validity status.

## Supported readers

| Artifact | Schema |
| --- | --- |
| Evaluation, sensitivity and robustness | `process_result/v1`, `process_result_bundle/v1` |
| Affine and storage controllers | `affine_controller/v1`, `storage_controller/v1` |
| Frozen controller binding | `frozen_controller/v1` |
| Equipment network | `process_network/v1` |
| Engineering change comparison | `engineering_change_comparison/v1` |
| Operating limit and dispatch sensitivity | `operating_limit/v1`, `dispatch_sensitivity/v1` |
| IDAES affine export | `idaes_linear_export/v1` |
| Shared commands and local covariance propagation | `idaes_shared_controls/v1`, `idaes_local_uncertainty/v1` |
| Calibration and joint bootstrap | `calibration_result/v1`, `calibration_bootstrap/v1` |
| Calibration diagnostics and held-out validation | `calibration_diagnostics/v1`, `calibration_validation/v1` |
| Original frozen heat contract | Integer schema `1` with its original `model_id` |

The allowlist is explicit. Unknown schemas are rejected, and document contents
cannot name Python classes, modules or callbacks to execute. Export-only reports,
such as controller synthesis and certificate-verification summaries, remain
export-only. Domain manifests use `domain_from_json` and the same strict parser.
Reading numerical integration artifacts does not require Pyomo, IDAES, pandas,
CVXPY or solver installations.

## Shared transport, separate validation

All artifact readers reject duplicate JSON keys at any depth, nonfinite literals
(`NaN`, `Infinity`, `-Infinity`), and numbers such as `1e999` that overflow the
floating-point representation. Envelope fields and schema versions are checked;
existing content identities are recomputed where the schema declares them.

Each family retains its own validation. Process results check evidence and
contract references; compact results resolve and check their artifact tables.
Calibration and uncertainty reports retain statistical and numerical scope
qualifications. Operating limits check derived bounds against their trials.
Frozen controllers reconstruct only their supported declarative bindings.

Loading establishes structural and content consistency. It does not solve the
physical model, prove a calibration's accuracy, or independently check a
certificate's mathematics. `verify_result` remains a separate operation with its
own supported certificate families. A valid calibration report cannot be passed
off as a verified process certificate through the generic loader.

## Compatibility and implementation boundary

Existing schemas, identity bodies, canonical SHA-256 encoding, optional-field
defaults, and JSON presentation are preserved. Compact result identities remain
those of the expanded result. The IDAES linear-export schema has no envelope
content ID; this refactor does not add one or claim that it provides one.

Transport mechanics live in `_artifact_json.py`, which has no model or result
dependencies. `artifact_io.py` contains the allowlisted schema routing. Typed
`from_dict` methods consume already decoded envelopes for nested loading without
an encode/parse round trip; they leave caller mappings unchanged. Result classes
retain their constructors and semantic checks. New schemas should reuse this
transport boundary and add a reader explicitly, without moving result meaning
into a universal result base class.
