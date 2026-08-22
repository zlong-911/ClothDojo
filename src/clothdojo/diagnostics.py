"""Installation diagnostics for the single ClothDojo environment."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Final

CORE_REQUIREMENTS: Final = {
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

TRAINING_REQUIREMENTS: Final = {
    "rlinf": "0.4.0",
    "rlinf-openpi": "0.1.1",
    "rlinf-transformer-openpi": "4.53.2",
    "hydra-core": "1.4.0.dev1",
    "omegaconf": "2.4.0.dev4",
    "lerobot": "0.1.0",
    "datasets": "3.6.0",
    "pyarrow": "20.0.0",
    "ml-dtypes": "0.4.1",
    "tensorstore": "0.1.74",
    "huggingface-hub": "0.32.3",
    "fsspec": "2025.3.0",
    "gcsfs": "2025.3.0",
    "zarr": "3.0.8",
    "opencv-python": "4.11.0.86",
    "opencv-python-headless": "4.11.0.86",
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "torchcodec": "0.4.0",
    "av": "15.0.0",
    "tyro": "1.0.13",
    "setuptools": "75.8.2",
}

TRAINING_CONFLICTS: Final = ("transformers",)


def inspect_environment(
    *,
    include_training: bool = False,
    smoke: bool = False,
) -> dict[str, object]:
    """Return installed versions and whether they match ClothDojo's baseline."""

    expected = dict(CORE_REQUIREMENTS)
    if include_training:
        expected.update(TRAINING_REQUIREMENTS)

    packages: dict[str, dict[str, object]] = {}
    compatible = True
    for distribution, required in expected.items():
        installed = _installed_version(distribution)
        matches = installed is not None and _matches(installed, required)
        packages[distribution] = {
            "installed": installed,
            "required": required,
            "matches": matches,
        }
        compatible = compatible and matches

    if include_training:
        for distribution in TRAINING_CONFLICTS:
            installed = _installed_version(distribution)
            matches = installed is None
            packages[distribution] = {
                "installed": installed,
                "required": "not installed",
                "matches": matches,
            }
            compatible = compatible and matches
    report: dict[str, object] = {"compatible": compatible, "packages": packages}
    if smoke:
        if compatible:
            smoke_report = run_newton_smoke()
        else:
            smoke_report = {
                "compatible": False,
                "skipped": "package versions do not match the required baseline",
            }
        report["smoke"] = smoke_report
        report["compatible"] = compatible and bool(smoke_report["compatible"])
    return report


def run_newton_smoke() -> dict[str, object]:
    """Exercise the Newton 1.5 build/collide/step path twice on CPU."""

    try:
        import newton
        import numpy as np
        import warp as wp

        def rollout() -> tuple[np.ndarray, np.ndarray]:
            newton.use_coord_layout_targets = True
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
            body = builder.add_body(
                xform=wp.transform(
                    wp.vec3(0.0, 0.0, 1.0),
                    wp.quat_identity(),
                )
            )
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            builder.add_ground_plane()
            model = builder.finalize(device="cpu", requires_grad=False)
            solver = newton.solvers.SolverXPBD(
                model,
                iterations=8,
                deterministic=wp.DeterministicMode.RUN_TO_RUN,
            )
            pipeline = newton.CollisionPipeline(model, deterministic=True)
            contacts = pipeline.contacts()
            state_0 = model.state()
            state_1 = model.state()
            control = model.control()
            for _ in range(4):
                for _ in range(2):
                    state_0.clear_forces()
                    pipeline.collide(state_0, contacts)
                    solver.step(
                        state_0,
                        state_1,
                        control,
                        contacts,
                        1.0 / 100.0,
                    )
                    state_0, state_1 = state_1, state_0
            return state_0.body_q.numpy(), state_0.body_qd.numpy()

        first_q, first_qd = rollout()
        second_q, second_qd = rollout()
        finite = bool(np.isfinite(first_q).all() and np.isfinite(first_qd).all())
        deterministic = bool(
            np.array_equal(first_q, second_q) and np.array_equal(first_qd, second_qd)
        )
        return {
            "compatible": finite and deterministic,
            "device": "cpu",
            "finite": finite,
            "deterministic": deterministic,
        }
    except Exception as error:
        return {
            "compatible": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _matches(installed: str, required: str) -> bool:
    try:
        installed_release = _numeric_release(installed)
        required_release = _numeric_release(required)
    except ValueError:
        return installed == required

    if len(required_release) == 2:
        return installed_release[:2] == required_release

    width = max(len(installed_release), len(required_release))
    return installed_release + (0,) * (width - len(installed_release)) == (
        required_release + (0,) * (width - len(required_release))
    )


def _numeric_release(value: str) -> tuple[int, ...]:
    public = value.partition("+")[0]
    return tuple(int(component) for component in public.split("."))
