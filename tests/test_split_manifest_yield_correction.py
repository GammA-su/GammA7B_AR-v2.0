import importlib.util
from pathlib import Path


def _load_module():
    mod_path = Path("scripts/33_make_split_manifest.py")
    spec = importlib.util.spec_from_file_location("make_split_manifest", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_yield_correction_renormalizes_and_shifts():
    m = _load_module()
    base = {"a": 0.5, "b": 0.5}
    yield_data = {"a": {"avg_yield": 10.0}, "b": {"avg_yield": 1.0}}
    corrected = m._apply_yield_correction(base, yield_data)
    normalized = m._normalized_weights(corrected)
    assert abs(sum(normalized.values()) - 1.0) < 1e-6
    assert normalized["a"] < 0.5
    assert normalized["b"] > 0.5
