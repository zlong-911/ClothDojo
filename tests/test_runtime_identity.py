from pathlib import Path

from clothdojo.runtime_identity import runtime_source_sha256


def test_runtime_identity_covers_python_paths_and_bytes(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("first\n", encoding="utf-8")
    before = runtime_source_sha256(tmp_path)

    (tmp_path / "a.py").write_text("second\n", encoding="utf-8")
    after_content = runtime_source_sha256(tmp_path)
    (tmp_path / "a.py").rename(tmp_path / "b.py")
    after_name = runtime_source_sha256(tmp_path)

    assert before != after_content
    assert after_content != after_name
