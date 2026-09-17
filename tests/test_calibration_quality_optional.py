"""Calibration quality records remain useful without optional modeling backends."""

from pathlib import Path
import subprocess  # nosec B404
import sys

import pytest


@pytest.mark.parametrize("kind", ["diagnostics", "validation"])
def test_quality_records_read_and_summarize_without_optional_backends(kind):
    payload = (
        Path(__file__)
        .with_name("fixtures")
        .joinpath(f"calibration_{kind}_v1.json")
        .read_text()
    )
    script = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'idaes', 'pyomo', 'pandas', 'highspy'}:
            raise ImportError('optional backend deliberately unavailable')
sys.meta_path.insert(0, BlockOptional())
from operange.integrations.parmest import CalibrationDiagnostics, CalibrationValidation
cls = CalibrationDiagnostics if sys.argv[1] == 'diagnostics' else CalibrationValidation
result = cls.from_json(sys.stdin.read())
if result.execution != 'completed' or cls.from_json(result.to_json()) != result:
    raise RuntimeError('missing or changed portable result')
if cls is CalibrationDiagnostics:
    if result.analysis['rank'] != 1 or len(result.analysis['weak_directions']) != 1:
        raise RuntimeError('missing local information analysis')
else:
    if result.metrics[0]['max_absolute_residual'] < 700:
        raise RuntimeError('missing held-out accuracy metrics')
if any(name.split('.')[0] in {'idaes', 'pyomo', 'pandas', 'highspy'} for name in sys.modules):
    raise RuntimeError('quality reader imported optional backends')
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, kind],
        input=payload,
        check=True,
        capture_output=True,
        text=True,
    )  # nosec B603
