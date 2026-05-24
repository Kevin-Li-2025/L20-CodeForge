from __future__ import annotations

import importlib.util
from pathlib import Path


def load_xcoder_builder_module():
    script = Path(__file__).parents[1] / "scripts" / "build_xcoder_sft_jsonl.py"
    spec = importlib.util.spec_from_file_location("build_xcoder_sft_jsonl", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_xcoder_row_to_sft_record() -> None:
    builder = load_xcoder_builder_module()

    record = builder.row_to_sft_record(
        {"query": "Solve A", "response": "```python\nprint(1)\n```", "source": "synthetic"},
        system_prompt="system",
        max_query_chars=None,
        max_response_chars=None,
        exclusion_hashes=set(),
        keep_metadata=True,
    )

    assert record["dataset"] == "IIGroup/X-Coder-SFT-376k"
    assert record["messages"][0] == {"role": "system", "content": "system"}
    assert record["messages"][1] == {"role": "user", "content": "Solve A"}
    assert record["messages"][2]["content"].endswith("\n")
    assert record["metadata"] == {"source": "synthetic"}


def test_xcoder_row_exclusion_uses_normalized_query_hash() -> None:
    builder = load_xcoder_builder_module()
    query = "  Solve   A\n"
    exclusion_hashes = {builder.stable_hash(builder.normalize_for_overlap(query))}

    record = builder.row_to_sft_record(
        {"query": "solve a", "response": "answer"},
        system_prompt="system",
        max_query_chars=None,
        max_response_chars=None,
        exclusion_hashes=exclusion_hashes,
        keep_metadata=False,
    )

    assert record is None
