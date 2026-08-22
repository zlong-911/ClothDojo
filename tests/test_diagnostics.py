from clothdojo import diagnostics


def test_environment_check_requires_exact_major_minor_baseline(monkeypatch) -> None:
    versions = {
        "newton": "1.5.0",
        "warp-lang": "1.16.0",
        "mujoco": "3.11.0",
        "mujoco-warp": "3.11.0",
        "usd-core": "26.3",
        "newton-usd-schemas": "0.4.1",
        "numpy": "1.26.4",
        "scipy": "1.17.1",
        "pyyaml": "6.0.3",
    }
    monkeypatch.setattr(diagnostics, "_installed_version", versions.get)

    report = diagnostics.inspect_environment()

    assert report["compatible"] is True


def test_environment_check_rejects_old_newton(monkeypatch) -> None:
    versions = {
        "newton": "1.4.0",
        "warp-lang": "1.16.0",
        "mujoco": "3.11.0",
        "mujoco-warp": "3.11.0",
        "usd-core": "26.3",
        "newton-usd-schemas": "0.4.1",
        "numpy": "1.26.4",
        "scipy": "1.17.1",
        "pyyaml": "6.0.3",
    }
    monkeypatch.setattr(diagnostics, "_installed_version", versions.get)

    report = diagnostics.inspect_environment()

    assert report["compatible"] is False
    assert report["packages"]["newton"]["matches"] is False


def test_training_check_rejects_overlapping_transformers_distribution(monkeypatch) -> None:
    versions = {
        "newton": "1.5.0",
        "warp-lang": "1.16.0",
        "mujoco": "3.11.0",
        "mujoco-warp": "3.11.0",
        "usd-core": "26.3",
        "newton-usd-schemas": "0.4.1",
        "numpy": "1.26.4",
        "scipy": "1.17.1",
        "pyyaml": "6.0.3",
        "rlinf": "0.4.0",
        "rlinf-openpi": "0.1.1",
        "rlinf-transformer-openpi": "4.53.2",
        "lerobot": "0.1.0",
        "datasets": "3.6.0",
        "torch": "2.7.1+cu128",
        "torchvision": "0.22.1+cu128",
        "torchcodec": "0.4.0",
        "av": "15.0.0",
        "tyro": "1.0.13",
        "setuptools": "75.8.2",
        "transformers": "4.53.2",
    }
    monkeypatch.setattr(diagnostics, "_installed_version", versions.get)

    report = diagnostics.inspect_environment(include_training=True)

    assert report["compatible"] is False
    assert report["packages"]["transformers"]["matches"] is False


def test_version_match_accepts_pep440_local_cuda_version() -> None:
    assert diagnostics._matches("2.7.1+cu128", "2.7.1") is True
    assert diagnostics._matches("0.5+cu128", "0.5.0") is True


def test_smoke_is_skipped_when_versions_do_not_match(monkeypatch) -> None:
    monkeypatch.setattr(diagnostics, "_installed_version", lambda distribution: None)

    report = diagnostics.inspect_environment(smoke=True)

    assert report["compatible"] is False
    assert report["smoke"] == {
        "compatible": False,
        "skipped": "package versions do not match the required baseline",
    }


def test_smoke_result_participates_in_environment_compatibility(monkeypatch) -> None:
    versions = {
        "newton": "1.5.0",
        "warp-lang": "1.16.0",
        "mujoco": "3.11.0",
        "mujoco-warp": "3.11.0",
        "usd-core": "26.3",
        "newton-usd-schemas": "0.4.1",
        "numpy": "1.26.4",
        "scipy": "1.17.1",
        "pyyaml": "6.0.3",
    }
    monkeypatch.setattr(diagnostics, "_installed_version", versions.get)
    monkeypatch.setattr(
        diagnostics,
        "run_newton_smoke",
        lambda: {"compatible": False, "error": "simulated failure"},
    )

    report = diagnostics.inspect_environment(smoke=True)

    assert report["compatible"] is False
    assert report["smoke"]["error"] == "simulated failure"
