import pytest

from gamma7b_data.cli import hf_configs_cmd


def test_hf_configs_wikipedia_returns_nonzero(capsys):
    pytest.importorskip("datasets")
    try:
        hf_configs_cmd(dataset="wikipedia", lang=None, script=None, contains=None, filter=None)
    except Exception as exc:
        pytest.skip(f"hf-configs requires online access in this test: {exc}")

    out = capsys.readouterr().out
    total_line = next((line for line in out.splitlines() if line.startswith("Total configs:")), None)
    assert total_line is not None
    total = int(total_line.split(":", 1)[1].strip())
    assert total > 0
