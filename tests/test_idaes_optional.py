"""The public binding declarations do not make IDAES a base dependency."""

# Use an isolated interpreter for a fixed import test.
import subprocess  # nosec B404
import sys


def test_idaes_import_and_missing_extra_are_explicit():
    script = """
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'idaes', 'pyomo', 'highspy'}:
            raise ImportError('optional dependency deliberately unavailable')

sys.meta_path.insert(0, BlockOptional())
from operange import Coordinate, ParameterSpace
from operange.integrations.idaes import IdaesAdapter, IdaesRequirement, IdaesVariable
if any(name.split('.')[0] in {'idaes', 'pyomo', 'highspy'} for name in sys.modules):
    raise RuntimeError('optional backend imported eagerly')
try:
    IdaesAdapter(model=object(), name='missing backend',
                 input_space=ParameterSpace((Coordinate('x', '1', 'test'),)),
                 inputs=(IdaesVariable('x', 'x', '1'),), controls=(),
                 outputs=(IdaesVariable('y', 'y', '1'),),
                 requirements=(IdaesRequirement('r', 'r', '1'),))
except ImportError as exc:
    if 'operange[idaes]' not in str(exc):
        raise
else:
    raise RuntimeError('missing optional dependency was not explained')
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script], check=True, capture_output=True, text=True
    )  # nosec B603
