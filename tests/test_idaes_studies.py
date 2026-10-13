"""Adapter-level typed studies retain the numerical IDAES evidence boundary."""

from dataclasses import replace

import pytest

from operange import OperatingSpec, SharedControlsRequest, UncertaintyPropagationRequest


def test_shared_request_preflight_is_read_only_and_does_not_solve(monkeypatch):
    from tests.test_idaes_shared_controls import boiler
    from operange.integrations import _idaes_execution
    from operange.integrations._idaes_model import model_manifest

    source, adapter, domain = boiler()
    request = SharedControlsRequest(
        domain, operating=OperatingSpec.shared(adapter.controls)
    )
    before = model_manifest(adapter._template), model_manifest(source)

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight prepared or solved a model")

    with monkeypatch.context() as patch:
        patch.setattr(_idaes_execution.PointExecution, "prepare", forbidden)
        patch.setattr(_idaes_execution, "solve", forbidden)
        report = adapter.check_study(request)
        assert report.supported and report.coverage == "finite_scenarios"  # nosec B101
        assert (
            "shared_commands" in report.evidence
            and "independent_certificate" not in report.evidence
        )  # nosec B101
        rejected = adapter.check_study(
            replace(request, required_evidence=("independent_certificate",))
        )
        assert rejected.status == "unsupported"  # nosec B101
        wrong = replace(
            request,
            operating=OperatingSpec.dispatch(adapter.controls, domain.space.names),
        )
        assert adapter.check_study(wrong).status == "invalid"  # nosec B101
    assert before == (model_manifest(adapter._template), model_manifest(source))  # nosec B101
    result = adapter.run_study(request)
    assert result.execution == "completed" and result.audit.payload.verdict == "pass"  # nosec B101
    assert result.commands == pytest.approx({"fuel": 10})  # nosec B101


def test_local_request_uses_the_same_parameter_and_response_validation(monkeypatch):
    from tests.test_idaes_uncertainty import boiler, PARAMETERS, COVARIANCE
    from operange.integrations import _idaes_execution
    from operange.integrations._idaes_model import model_manifest

    source, adapter = boiler()
    request = UncertaintyPropagationRequest(
        PARAMETERS,
        COVARIANCE,
        OperatingSpec.fixed(adapter.controls, {"fuel": 15}),
        outputs=("steam",),
    )
    before = model_manifest(source), model_manifest(adapter._template)

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight solved or initialized")

    with monkeypatch.context() as patch:
        patch.setattr(_idaes_execution, "solve", forbidden)
        patch.setattr(_idaes_execution.PointExecution, "prepare", forbidden)
        cap = adapter.check_study(request)
        assert cap.supported and cap.coverage == "local"  # nosec B101
        assert (
            "local_covariance" in cap.evidence
            and "continuous_coverage" not in cap.evidence
        )  # nosec B101
        assert (
            adapter.check_study(replace(request, outputs=("missing",))).status
            == "invalid"
        )  # nosec B101
        assert (
            adapter.check_study(replace(request, covariance=((1, 2), (2, 1)))).status
            == "invalid"
        )  # nosec B101
        assert not adapter.check_study(
            replace(request, required_evidence=("statistical_coverage",))
        ).supported  # nosec B101
    assert before == (model_manifest(source), model_manifest(adapter._template))  # nosec B101
    result = adapter.run_study(request)
    assert result.execution == "completed" and result.mode == "fixed"  # nosec B101
    assert result.jacobian[0] == pytest.approx((15, -1))  # nosec B101
