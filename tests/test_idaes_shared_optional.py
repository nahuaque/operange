"""Saved shared-control evidence needs no optional modeling or solver backend."""

from pathlib import Path
import subprocess  # nosec B404
import sys

from operange.integrations.idaes import IdaesSharedControlResult


def test_shared_control_result_is_portable_without_optional_backends():
    payload = (
        Path(__file__)
        .with_name("fixtures")
        .joinpath("idaes_shared_controls_v1.json")
        .read_text()
    )
    result = IdaesSharedControlResult.from_json(payload)
    if result.execution != "completed":
        raise AssertionError("fixture must retain checked commands")
    script = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'idaes', 'pyomo', 'pandas', 'highspy'}:
            raise ImportError('optional backend deliberately unavailable')
sys.meta_path.insert(0, BlockOptional())
from operange.integrations.idaes import IdaesSharedControlResult
result = IdaesSharedControlResult.from_json(sys.stdin.read())
if result.execution != 'completed' or result.commands['fuel'] != 10:
    raise RuntimeError('missing checked commands')
if result.objective_value != 10 or result.audit.payload.verdict != 'pass':
    raise RuntimeError('missing objective or audit')
if result.scope['optimality'] != 'not_certified':
    raise RuntimeError('unsupported optimality claim')
if IdaesSharedControlResult.from_json(result.to_json()) != result:
    raise RuntimeError('round trip changed result')
if any(name.split('.')[0] in {'idaes', 'pyomo', 'pandas', 'highspy'} for name in sys.modules):
    raise RuntimeError('shared-control reader imported an optional backend')
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        input=payload,
        check=True,
        capture_output=True,
        text=True,
    )  # nosec B603
