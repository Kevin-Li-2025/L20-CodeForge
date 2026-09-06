from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest


def module():
    path = Path(__file__).parents[1] / "scripts/verify_rlvr_v2_receipts.py"
    spec = importlib.util.spec_from_file_location("verify_rlvr_v2_receipts", path)
    assert spec and spec.loader
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_published_receipts_agree_without_claiming_model_success():
    verifier = module()
    result = verifier.verify(verifier.DEFAULT_PACKAGE)
    assert result["receipt_consistency"] == "PASS"
    assert result["model_quality_target_achieved"] is False


@pytest.mark.parametrize(
    "relative,keys,value",
    [
        ("campaign_summary.json", ["target_achieved"], True),
        ("campaign_summary.json", ["development", "seed42", "rlvr_rstar_passed"], 81),
        ("campaign_summary.json", ["selected_candidate"], "seed44"),
        ("campaign_summary.json", ["report_only_final", "evalplus", "mbppplus", "delta_tasks"], 0),
        ("receipts/development/base-new-dev.report.json", ["output_sha256"], "0" * 64),
        ("receipts/development/sft-seed44-new-dev.report.json", ["summary", "samples"], 199),
        ("receipts/comparisons/rlvr-selection.json", ["required_target_passes"], 79),
        ("receipts/final/evalplus-no-regression.json", ["evalplus_no_regression_passed"], True),
    ],
)
def test_mutated_receipt_fails_closed(tmp_path, relative, keys, value):
    verifier = module()
    package = tmp_path / "package"
    shutil.copytree(verifier.DEFAULT_PACKAGE, package)
    path = package / relative
    document = json.loads(path.read_text())
    node = document
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        verifier.verify(package)
