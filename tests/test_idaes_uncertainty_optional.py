"""Local covariance records can be inspected with the base installation."""

from pathlib import Path
import subprocess  # nosec B404
import sys


def test_local_uncertainty_record_reads_without_optional_backends():
    payload = (
        Path(__file__)
        .with_name("fixtures")
        .joinpath("idaes_local_uncertainty_v1.json")
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
from operange.integrations.idaes import IdaesUncertaintyResult
r = IdaesUncertaintyResult.from_json(sys.stdin.read())
if abs(r.statistics['responses'][0]['variance'] - 1.3) > 1e-10:
    raise RuntimeError('correlation lost')
if IdaesUncertaintyResult.from_json(r.to_json()) != r:
    raise RuntimeError('round trip changed evidence')
if any(n.split('.')[0] in {'idaes', 'pyomo', 'pandas', 'highspy'} for n in sys.modules):
    raise RuntimeError('optional backend imported')
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        input=payload,
        check=True,
        capture_output=True,
        text=True,
    )  # nosec B603
