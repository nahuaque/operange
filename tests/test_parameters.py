"""Provider-independent values and moments keep coordinate identity explicit."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess  # nosec B404
import sys

import numpy as np
import pytest

from operange import Coordinate, ParameterSpace, Scenario
from operange.parameters import (
    ParameterBinding,
    ParameterCovariance,
    ParameterPoint,
    ParameterSamples,
)
from operange.integrations.parmest import CalibrationBootstrap


def space():
    return ParameterSpace(
        (
            Coordinate("eta", "1", "External estimates", nominal=0.8),
            Coordinate("loss", "MW", "External estimates", nominal=1),
        )
    )


def test_samples_keep_whole_vectors_duplicates_labels_and_native_order():
    sample = ParameterSamples(
        space(),
        (
            Scenario("a", {"loss": 1, "eta": 0.8}, "External sample"),
            Scenario("b", {"eta": 0.6, "loss": 2}, "External sample"),
            Scenario("duplicate", {"eta": 0.8, "loss": 1}, "Repeated vector"),
        ),
        {"source": "external-estimator"},
    )
    values = np.asarray([[0.8, 1], [0.6, 2], [0.8, 1]])
    np.testing.assert_allclose(
        sample.statistics["covariance"], np.cov(values.T, ddof=1)
    )
    np.testing.assert_allclose(sample.statistics["mean"], values.mean(axis=0))
    assert sample.as_domain().scenarios == sample.samples  # nosec B101
    assert sample.covariance.provenance["source"] == "external-estimator"  # nosec B101
    assert sample.covariance.provenance["samples"] == 3  # nosec B101
    assert ParameterSamples(**json.loads(json.dumps(sample.to_dict()))) == sample  # nosec B101


def test_covariance_aligns_by_name_and_unit_without_inventing_a_distribution():
    cov = ParameterCovariance(space(), ((0.01, 0.04), (0.04, 0.25)))
    reversed_space = ParameterSpace(tuple(reversed(space().coordinates)))
    np.testing.assert_allclose(
        cov.in_space(reversed_space), [[0.25, 0.04], [0.04, 0.01]]
    )
    with pytest.raises(ValueError, match="units"):
        cov.in_space(
            ParameterSpace(
                (space().coordinates[0], replace(space().coordinates[1], unit="kW"))
            )
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        ParameterCovariance(space(), ((1e-30, 2e-15), (2e-15, 1)))


def test_parameter_records_snapshot_inputs_and_validate_identity():
    values = {"eta": 0.8, "loss": 1}
    point = ParameterPoint(space(), values, {"source": "test"})
    values["eta"] = 0
    assert point.values["eta"] == 0.8  # nosec B101
    assert ParameterPoint(**point.to_dict()) == point  # nosec B101
    with pytest.raises(ValueError, match="exactly"):
        ParameterPoint(space(), {"eta": 0.8})
    with pytest.raises(ValueError):
        ParameterBinding("eta", "", "1")
    one = ParameterSamples(space(), (Scenario("only", point.values, "external"),))
    assert len(one.as_domain().scenarios) == 1  # nosec B101
    with pytest.raises(ValueError, match="two"):
        _ = one.statistics
    with pytest.raises(ValueError, match="duplicate"):
        replace(one, samples=(one.samples[0], one.samples[0]))


def test_calibration_converters_keep_saved_records_and_domain_identity():
    saved = (
        Path(__file__)
        .with_name("fixtures")
        .joinpath("calibration_bootstrap_v1.json")
        .read_text()
    )
    bootstrap = CalibrationBootstrap.from_json(saved)
    identity = bootstrap.result_id
    point = bootstrap.calibration.parameter_point
    samples = bootstrap.parameter_samples
    assert point.values == {
        p.component: p.value for p in bootstrap.calibration.estimates
    }  # nosec B101
    assert samples.as_domain() == bootstrap.as_domain()  # nosec B101
    assert samples.statistics == bootstrap.statistics  # nosec B101
    assert bootstrap.result_id == identity  # nosec B101
    assert CalibrationBootstrap.from_json(bootstrap.to_json()) == bootstrap  # nosec B101
    assert samples.provenance["uncertainty_ref"] == bootstrap.ref.to_dict()  # nosec B101


def test_neutral_records_do_not_import_modeling_or_estimation_backends():
    script = """
import importlib.abc
import sys
class BlockBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('operange.integrations') or fullname.split('.')[0] in {'idaes','pyomo','pandas','highspy','cvxpy'}:
            raise ImportError('backend deliberately blocked')
sys.meta_path.insert(0, BlockBackend())
from operange import ParameterPoint, ParameterCovariance, ParameterSamples, ParameterBinding, ParameterSpace, Coordinate, Scenario
space = ParameterSpace((Coordinate('p','1','external'),))
point = ParameterPoint(space, {'p':2})
samples = ParameterSamples(space, (Scenario('a',{'p':1},'external'),Scenario('b',point.values,'external')))
if samples.statistics['covariance'] != ((0.5,),):
    raise RuntimeError('incorrect sample covariance')
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script], check=True, capture_output=True, text=True
    )  # nosec B603
