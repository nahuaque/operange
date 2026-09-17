"""Calibration evidence is readable with only Operange's base dependencies."""

from pathlib import Path
import subprocess  # nosec B404
import sys

from operange.integrations.parmest import CalibrationResult


def test_portable_calibration_without_optional_backends():
    payload = (
        Path(__file__)
        .with_name("fixtures")
        .joinpath("calibration_result_v1.json")
        .read_text()
    )
    result = CalibrationResult.from_json(payload)
    if result.execution != "completed":
        raise AssertionError("fixture must retain checked fit")
    script = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'idaes', 'pyomo', 'pandas', 'highspy'}:
            raise ImportError('optional backend deliberately unavailable')
sys.meta_path.insert(0, BlockOptional())
from operange.integrations.parmest import CalibrationResult, ParmestCalibration
result = CalibrationResult.from_json(sys.stdin.read())
if abs(result.estimates[0].value - 0.7425) > 1e-8:
    raise RuntimeError('wrong estimate')
if abs(result.experiments[0].observations[0].standardized_residual + 0.75) > 1e-6:
    raise RuntimeError('missing residual')
if CalibrationResult.from_json(result.to_json()) != result:
    raise RuntimeError('round trip changed result')
if any(name.split('.')[0] in {'idaes', 'pyomo', 'pandas', 'highspy'} for name in sys.modules):
    raise RuntimeError('optional backend imported eagerly')
try:
    ParmestCalibration(name='missing backend', experiments={'run': object()})
except ImportError as exc:
    if 'operange[idaes]' not in str(exc):
        raise
else:
    raise RuntimeError('missing backend was not explained')
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        input=payload,
        check=True,
        capture_output=True,
        text=True,
    )  # nosec B603
