# Independently verify exported certificates

`verify_result()` checks exported engineering conclusions without running an
optimizer or calling the model's evaluation or audit methods. It first validates
the result structure, content identities and references, then reconstructs the
physical problem from the saved declarations and checks its certificates with
separate exact rational arithmetic.

```python
from pathlib import Path
from operange import verify_result

report = verify_result(
    Path("audit.json").read_text(),
    expected_contract_id=original_claim.contract.ref.artifact_id,
)
print(report.status)
for check in report.checks:
    print(check.path, check.status, check.message)
```

The input can be standard or compact result JSON, UTF-8 bytes, an exported
dictionary, or a typed result. `expected_contract_id` is optional: supplying it
pins the declaration that you expect. Without it, verification is relative to
the model, domain, requirements and permissions included in the export.
Content identities do not authenticate the author or establish the accuracy of
the physical model.

`result_from_json()` retains its existing role: validating structure and
identities without re-proving evidence. A modified certificate can pass that
loader if its identities are recomputed. The independent verifier must still
reject incorrect arithmetic or evidence for a different physical problem.

## Supported checks

The first implementation covers declarative `LinearProcessAdapter` results
over boxes, finite sets and explicit convex hulls:

| Certificate | Independent checks |
| --- | --- |
| Feasible dispatch | Domain membership, fixed commands, hard control bounds, every operating limit and selected requirement, physical outputs and directed residual rounding. |
| Linear infeasibility | Reconstructed normalized rows and coordinates, nonnegative multipliers, exact weighted rows and right-hand side, and a strictly positive bounded-box contradiction. |
| Operating objective | Declared affine or scaled quadratic objective, physical attained value, separable Lagrangian lower bound, exported bounds, and any claimed optimality gap. |
| Conflict explanation | The declared row subset and background bounds, its contradiction, and a feasible command witness for every removal claimed to prove irreducibility. |
| Single or joint equipment relief | Permitted changes, units and scales, augmented rows, objective bounds, changed physical limits, candidate commands and any minimum or unreachability claim. |
| Shared equipment relief | Every scenario block, one common change vector, its bounds, and recursive verification of the embedded changed audit with the proposed commands. |
| Finite and vertex audits | Complete scenario/generator identities and references, supporting responses, and the failure witness or convex-combination coverage argument. |
| Frozen affine controllers | Allowed observations, exact affine execution followed by command rounding, physical replay, and continuous box/hull residual enclosures. |

The verifier independently reconstructs rows from the original model. A valid
contradiction for an altered matrix, a missing equipment constraint, or a relief
plan that quietly changes fixed commands cannot establish the original claim.
Solver logs and solver termination statuses are not mathematical certificates.

For a continuous box or hull, vertex dispatch coverage establishes existence of
real-valued adjustable commands between generators. Verifying that certificate
does not turn it into an executable controller. Frozen-controller verification
separately checks the effect of rounding its prescribed commands throughout the
domain.

Hull membership is currently checked at explicitly declared generators. A
point result at another hull member needs an independent barycentric membership
checker and is reported as `unsupported`. Other model families, sensitivities,
distance/threshold searches, and support proofs for other convex geometries are
also outside this version's scope. Unknown evidence is reported explicitly.

For synthesis, pass the exported `search.audit` to `verify_result()`. The
`controller_synthesis/v1` report and its separate performance enclosure are not
inputs to this result verifier. The controller's physical audit is checkable;
synthesis still makes no certified optimality claim across controllers.

## Read the report

`CertificateVerification` contains the checked result's `artifact_ref` and
individual `VerificationCheck` records. Its `status` is:

| Status | Meaning |
| --- | --- |
| `verified` | All applicable supported checks passed. |
| `failed` | A declaration, physical response or mathematical certificate disagrees with its independently reconstructed check. |
| `unsupported` | A required proof family, incomplete conclusion or verification limit prevents a complete check. |
| `invalid` | Input JSON, structure, identities or verification options are invalid. |

`verified` is true only for the first status. Failures take precedence over
unsupported checks. A partial collection of passing checks never becomes a
verified report. Verification of valid bounded objective evidence does not
claim that its gap has closed; the original resolution remains authoritative
only to the extent independently checked.

`report.to_json()` exports `certificate_verification/v1`, including diagnostic
paths into the checked artifact. Narrative provenance, diagnostic prose and
numerical solver logs are retained by the original artifact but are not
authenticated by mathematical verification.

## Verify from the command line

The installed package provides a module command; CVXPY is not required:

```bash
python -m operange.verify audit.json
python -m operange.verify audit.json --expected-contract-id 'sha256:...'
cat audit.json | python -m operange.verify -
```

The command prints the JSON verification report. Exit codes are `0` for
verified, `1` for failed mathematics or a mismatched pinned contract, and `2`
for invalid or unsupported inputs. It reads local files or standard input and
does not access a solver or the network.

Default limits are 32 MB of JSON, 10,000 report checks, 10,000 enumerated
generators, and 16 nested verification levels. `max_bytes` and `max_checks` are
API options; `--max-bytes` is also available on the command. Exact rational
strings have bounded lengths and exponents. Box controller extrema are checked
directly from coordinate bounds without enumerating corners.

The [consumer example](https://github.com/nahuaque/operange/blob/main/examples/certificate_verification.py)
verifies a shared heater-capacity plan and shows that a freshly rehashed but
incorrect infeasibility certificate is rejected:

```bash
uv run python -m examples.certificate_verification
```
