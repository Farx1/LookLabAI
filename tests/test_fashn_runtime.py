"""FASHN runtime plumbing. Real inference lives in test_gpu.py (skipped without CUDA)."""

import importlib.util
import sys

import pytest

from looklab_ml.tryon import FashnRuntime
from looklab_ml.tryon.fashn_runtime import _DiscardedParser, _install_parser_interface_shim


def test_availability_reasons(tmp_path):
    rt = FashnRuntime(weights_dir=str(tmp_path))
    ok, why = rt.available()
    assert not ok
    if importlib.util.find_spec("fashn_vton") is None:
        assert "not installed" in why
    else:
        assert "weights missing" in why or "CUDA" in why


def test_parser_free_mode_accepts_only_parser_independent_inputs():
    rt = FashnRuntime(weights_dir="/nonexistent", parser_mode="parser_free")
    rt.validate("flat-lay", segmentation_free=True)
    with pytest.raises(ValueError, match="non-commercial"):
        rt.validate("model")
    with pytest.raises(ValueError):
        rt.validate("flat-lay", segmentation_free=False)
    FashnRuntime(weights_dir="/x", parser_mode="full").validate("model")


def test_interface_shim_has_no_model(monkeypatch):
    if importlib.util.find_spec("fashn_human_parser") is not None:
        pytest.skip("real fashn-human-parser installed; shim not used")
    monkeypatch.delitem(sys.modules, "fashn_human_parser", raising=False)
    _install_parser_interface_shim()
    mod = sys.modules["fashn_human_parser"]
    assert mod.__looklab_shim__ and mod.CATEGORY_TO_BODY_COVERAGE["one-pieces"] == "full"
    with pytest.raises(RuntimeError):
        mod.FashnHumanParser()
    np = pytest.importorskip("numpy")
    assert _DiscardedParser().predict(np.zeros((8, 6, 3), dtype=np.uint8)).shape == (8, 6)
    sys.modules.pop("fashn_human_parser")
