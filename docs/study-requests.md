# Study requests and capabilities

Use a typed request to inspect a particular study before running it. A `Claim`
already supplies the model, uncertainty domain, selected requirements and
operating permissions. The request supplies the operation, numerical settings
and required evidence capabilities.

```python
from operange import AuditRequest

request = AuditRequest(
    backend="scipy",
    max_vertices=256,
    required_evidence=("continuous_coverage", "independent_certificate"),
)
capability = claim.check_study(request)
if capability.supported:
    audit = claim.run_study(request)
else:
    print(capability.status, capability.scope)
```

`StudyCapability` reports `supported`, `unsupported` or `invalid`, with a reason
or scope, coverage, evidence capabilities, optional dependencies and limitations.
An invalid request has malformed settings, unknown coordinates or inconsistent
declarations. An unsupported request asks for a method, policy, coverage or
evidence that the binding cannot provide, or needs a missing optional dependency.

Preflight does not initialize, solve or enumerate scenarios. It checks static
declarations, options, units, permissions and resource limits. IDAES parameter
validation can inspect and clone the bound model; it does not change that model.
Dependency discovery does not import CVXPY or invoke solver executables.
Membership, numerical solver availability, baseline feasibility, local regularity
and successful proof construction remain execution checks.

Run `uv run python -m examples.study_requests` for a complete public-API consumer
that inspects support, audits a utility model and independently verifies its
certificate. It also shows a refused generator budget and shared-policy synthesis
capability without launching a controller search.

## Request types

All types are available from `operange`. Request mappings and sequences are copied
into immutable containers. For the original five queries, `None` option fields
mean “use this adapter's existing default.” Unknown constructor keywords fail
immediately; other invalid declarations can be inspected through preflight.

| Target | Request | Existing entry point |
| --- | --- | --- |
| `Claim` | `EvaluationRequest(realization, ...)` | `evaluate_result(...)` |
| `Claim` | `SensitivityRequest(realization, ...)` | `sensitivity_result(...)` |
| `Claim` | `AuditRequest(...)` | `audit_result(...)` |
| `Claim` | `BoundaryRequest(...)` | `boundary_result(...)` |
| `Claim` | `BreakingRequest(...)` | `breaking_result(...)` |
| `Claim` | `ControllerSynthesisRequest(...)` | `synthesize_controller(...)` |
| `Claim` | `OperatingLimitRequest(parameter, ...)` | `operating_limit(...)` |
| `Claim` | `DispatchSensitivityRequest(realization, changes, ...)` | `dispatch_sensitivity(...)` |
| `IdaesAdapter` | `SharedControlsRequest(domain, operating=...)` | `optimize_shared_controls(...)` |
| `IdaesAdapter` | `UncertaintyPropagationRequest(parameters, covariance, operating, ...)` | `propagate_uncertainty(...)` |

`check_study(target, request)` and `run_study(target, request)` are equivalent
function forms. The typed execution API raises `StudyRequestError` before
execution for an invalid or unsupported request. Its `capability` attribute
contains the report. A supported run returns the existing result family, which
can still report unresolved execution or incomplete evidence. The original
methods retain their existing result and exception conventions.

The five fields of `claim.capabilities` remain a compatible coarse overview.
They do not describe the newer studies or all option combinations. For example,
a shared-command linear claim can support controller synthesis while refusing
independent point dispatch. A box with a linear controller objective uses full-box
support without enumerating corners, so its `max_vertices` setting does not limit
that formulation. Tracking synthesis uses complete generators and respects that
limit. A controller with algebraic states supports finite synthesis but cannot
claim continuous rounded-controller coverage.

## Evidence requirements

`required_evidence` selects **method capabilities**, not promised outcomes. Check
the returned result's execution, qualifications and evidence before drawing an
engineering conclusion. Requiring `independent_certificate` makes a numerical-only
IDAES study unsupported; it does not turn numerical residual checks into proof.

`EvidenceRequirement` names the supported vocabulary:

| Name | Meaning |
| --- | --- |
| `checked_feasibility` | Checked response or dispatch against the declared physical constraints. |
| `infeasibility_certificate` | Exact bounded-linear contradiction when one is found. |
| `objective_bounds` | Bounds on an individual dispatch objective. |
| `continuous_coverage` | The method can cover the declared continuum, not just listed scenarios. |
| `controller_audit` | Separate checking of a saved controller's executed commands. |
| `local_sensitivity` | Qualified local response or objective-value derivatives. |
| `local_covariance` | Local propagation of an explicit parameter covariance. |
| `shared_commands` | One command vector checked across all declared scenarios. |
| `distance_bounds`, `range_bounds`, `relief_bounds` | The corresponding study's checked bounds. |
| `independent_certificate` | An existing certificate path supports independent result verification. |
| `global_optimality`, `statistical_coverage` | Stronger requirements that this request layer currently declines. |

Finite coverage, continuous coverage, local information, numerical source checks
and independent certificates remain distinct. Controller synthesis does not
advertise a global policy optimum, and local covariance propagation does not
advertise a nonlinear confidence region.

## IDAES operating studies

Use the adapter as the target for shared commands and local propagation. Solver
and initializer settings remain on the immutable adapter binding.

```python
from operange import (
    OperatingSpec, SharedControlsRequest, UncertaintyPropagationRequest,
)

shared = SharedControlsRequest(
    scenarios,
    operating=OperatingSpec.shared(adapter.controls),
    required_evidence=("shared_commands", "checked_feasibility"),
)
report = adapter.check_study(shared)

local = UncertaintyPropagationRequest(
    parameters=parameter_bindings,  # Ordered IdaesVariable declarations.
    covariance=parameter_covariance,  # ParameterCovariance or a matrix tuple.
    operating=OperatingSpec.fixed(adapter.controls, commands),
    outputs=("steam",),
)
report = adapter.check_study(local)
```

Omitting `SharedControlsRequest.operating` keeps the existing shared-command
default. Local requests require an explicit `OperatingSpec`; fully observed
reoptimization also needs native-unit perturbation steps. Partial fixed/free
local response and silently expanded observation permissions remain unsupported.

## Implementation boundary

Requests and reports live in `studies.py`; dispatch and static adapter checks are
private infrastructure. Newer execution paths share their preparation validators
with preflight. Built-in adapters expose `query_options(claim, operation)`, used
both by request inspection and their original option validation.

Existing consumer adapters still implement the original `ModelAdapter` protocol.
Their default requests use their declared operation support; structured evidence
capabilities are not inferred from their prose. Requests with options require an
explicit option declaration.

`_domain_generators` shares counting and deterministic traversal of finite sets,
box endpoints and hull vertices. Audit and synthesis retain separate coverage
arguments and caller-specific provenance. Enumeration itself establishes no
continuous coverage. The [independent verifier](certificate-verification.md)
continues to reconstruct its own proof inputs; requests do not replace results or
introduce a new result schema.
