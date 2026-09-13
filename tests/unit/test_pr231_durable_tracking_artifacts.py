from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _module() -> Any:
    path = Path(__file__).parents[2] / "tests/e2e/test_global_regime_v4_subproofs.py"
    spec = importlib.util.spec_from_file_location("pr231_subproofs_unit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PR-231 sub-proof module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tracking_artifacts_are_copied_to_a_durable_bundle(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "temporary" / "global_v4_plot_manifest.json"
    source.parent.joinpath("plots").mkdir(parents=True)
    source.parent.joinpath("plots", "quality.png").write_bytes(b"png")
    source.write_text(
        json.dumps({"entries": [{"png_path": "plots/quality.png"}]}),
        encoding="utf-8",
    )

    durable = module._persist_tracking_artifacts(source, tmp_path / "bundle")

    assert durable == tmp_path / "bundle" / "tracking-and-plots" / source.name
    assert durable.is_file()
    assert (durable.parent / "plots" / "quality.png").read_bytes() == b"png"
