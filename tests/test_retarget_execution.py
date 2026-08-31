"""Execution optimizations must retain physics boundaries and source ordering."""

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import clothdojo.data.retarget as module
from clothdojo.sim.runtime import ShirtFlattenRuntime


def test_single_candidate_reads_only_fields_used_by_the_objective():
    from clothdojo.data.retarget_dynamics import RobotDynamicsRetargeter

    retargeter = RobotDynamicsRetargeter.__new__(RobotDynamicsRetargeter)
    calls = []
    state = (np.array([1.]), np.array([2.]))
    command = np.array([3.])
    predicted = {"joint_q": np.array([4.]), "body_q": np.array([5.])}

    def read(fields):
        assert fields == ("joint_q", "body_q")
        calls.append("read")
        return predicted

    retargeter.runtime = SimpleNamespace(
        restore_rigid_state=lambda q, qd: calls.append((q, qd)), rigid_state=read,
    )
    retargeter._step = lambda values: calls.append(values)
    assert retargeter._trial(state, command) is predicted
    assert calls[0] == state
    assert calls[1] is command
    assert calls[2] == "read"


@pytest.mark.parametrize("graph", [False, True])
def test_inverse_dynamics_refreshes_state_and_acceleration_and_caches_layout(monkeypatch, graph):
    import mujoco_warp
    import warp as wp

    runtime = ShirtFlattenRuntime.__new__(ShirtFlattenRuntime)
    runtime.model = SimpleNamespace(joint_dof_count=6, device=wp.get_device("cpu"))
    runtime.spec = SimpleNamespace(control_fps=50, sim_substeps=6)
    runtime.state_0 = SimpleNamespace(joint_q=wp.zeros((2, 3), device="cpu"))
    runtime._rigid_step_graph = object() if graph else None
    runtime._inverse_graph = object() if graph else None
    runtime._inverse_layout = None
    runtime._inverse_acceleration = None
    reads = []

    def mapping(name, values):
        def numpy():
            reads.append(name)
            return np.asarray(values)
        return SimpleNamespace(numpy=numpy)

    data = SimpleNamespace(
        nworld=2, qacc=wp.zeros((2, 3), device="cpu"),
        qpos=wp.zeros((2, 3), device="cpu"),
        qfrc_inverse=wp.zeros((2, 3), device="cpu"),
        qfrc_constraint=wp.ones((2, 3), device="cpu"),
    )
    runtime.rigid = SimpleNamespace(
        use_mujoco_cpu=False, mjw_data=data,
        mjw_model=SimpleNamespace(opt=SimpleNamespace(timestep=wp.zeros(1, device="cpu"))),
        mj_model=SimpleNamespace(actuator_trnid=np.array([[1, 0], [0, 0]]),
                                 jnt_dofadr=np.array([0, 1, 2])),
        mjc_actuator_to_newton_idx=mapping("actuator", [0, 2]),
        mjc_dof_to_newton_dof=mapping("dof", [[2, 0, 1], [5, 3, 4]]),
        _update_mjc_data=lambda d, m, s: wp.copy(d.qpos, s.joint_q),
    )
    monkeypatch.setattr(mujoco_warp, "inverse", lambda m, d: d.qfrc_inverse.assign(
        2 * d.qacc.numpy() + d.qpos.numpy(),
    ))
    launches = []

    def launch(value):
        assert value is runtime._inverse_graph
        launches.append(value)
        runtime._evaluate_inverse_dynamics()

    monkeypatch.setattr(wp, "capture_launch", launch)
    for scale in (1., 2.):
        runtime.state_0.joint_q.fill_(scale)
        desired = scale * np.array([[10., 20., 30.], [-10., -20., -30.]])
        actual = runtime.inverse_dynamics(desired)
        expected = np.zeros_like(desired)
        expected[:, [0, 2]] = 2 * desired[:, [0, 2]] + scale + 1
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(data.qacc.numpy(), desired[:, [2, 0, 1]])
    assert reads == ["actuator", "dof"]
    assert len(launches) == (2 if graph else 0)
    np.testing.assert_allclose(runtime.rigid.mjw_model.opt.timestep.numpy(), [0.02 / 6])
    for invalid in (np.zeros(3), np.full((2, 3), np.nan)):
        with pytest.raises(ValueError, match="finite"):
            runtime.inverse_dynamics(invalid)


def test_rigid_restore_graph_reads_each_new_boundary(monkeypatch):
    import warp as wp

    runtime = ShirtFlattenRuntime.__new__(ShirtFlattenRuntime)
    runtime.model = SimpleNamespace(world_count=2, joint_coord_count=4, joint_dof_count=4)
    runtime._rigid_restore_graph = object()
    runtime._restore_joint_q = wp.zeros(4, device="cpu")
    runtime._restore_joint_qd = wp.zeros(4, device="cpu")
    restored = []

    def launch(graph):
        assert graph is runtime._rigid_restore_graph
        restored.append((runtime._restore_joint_q.numpy().copy(),
                         runtime._restore_joint_qd.numpy().copy()))

    monkeypatch.setattr(wp, "capture_launch", launch)
    for value in (1., 2.):
        runtime.restore_rigid_state_batch(np.full((2, 2), value), np.full((2, 2), -value))
    np.testing.assert_array_equal(restored[0][0], [1.] * 4)
    np.testing.assert_array_equal(restored[1][1], [-2.] * 4)
    with pytest.raises(ValueError, match="finite"):
        runtime.restore_rigid_state_batch(np.full((2, 2), np.nan), np.zeros((2, 2)))
    assert len(restored) == 2


@pytest.mark.parametrize("swaps", [False, True])
@pytest.mark.parametrize("steps", [0, 3])
def test_rigid_sequence_retains_initial_state_world_order_and_all_boundaries(swaps, steps):
    import warp as wp

    runtime = ShirtFlattenRuntime.__new__(ShirtFlattenRuntime)
    runtime.model = SimpleNamespace(
        world_count=2, joint_coord_count=4, joint_dof_count=4, body_count=2,
        device=wp.get_device("cpu"),
    )
    runtime._arm_action_count = 1
    runtime.state_0 = SimpleNamespace(
        joint_q=wp.zeros(4, device="cpu"), body_q=wp.zeros(2, dtype=wp.transform, device="cpu"),
    )
    runtime.state_1 = SimpleNamespace(
        joint_q=wp.zeros(4, device="cpu"), body_q=wp.zeros(2, dtype=wp.transform, device="cpu"),
    )
    calls = []

    def step(arm, grip):
        values = runtime.state_0.joint_q.numpy().reshape(2, 2)
        values = values + np.concatenate((arm.numpy(), grip.numpy()[:, :1]), axis=1)
        if swaps:
            runtime.state_0, runtime.state_1 = runtime.state_1, runtime.state_0
        runtime.state_0.joint_q.assign(values.ravel())
        calls.append(values.copy())

    runtime.step_rigid = step
    arms = np.arange(steps * 2, dtype=np.float32).reshape(steps, 2, 1)
    grips = np.ones((steps, 2, 2), dtype=np.float32)
    result = runtime.rollout_rigid(arms, grips, fields=("joint_q", "body_q"))
    assert set(result) == {"joint_q", "body_q"}
    assert result["body_q"].shape == (steps + 1, 2, 1, 7)
    expected = np.concatenate((np.zeros((1, 2, 2)),
        np.cumsum(np.concatenate((arms, grips[..., :1]), axis=-1), axis=0)), axis=0)
    np.testing.assert_array_equal(result["joint_q"], expected)
    assert len(calls) == steps
    with pytest.raises(ValueError, match="commands"):
        runtime.rollout_rigid(arms, np.zeros((steps + 1, 2, 2)))


@pytest.mark.parametrize("mask", [[False, False], [True, False], [True, True]])
@pytest.mark.parametrize("include_up", [False, True])
def test_batched_residual_matches_original_scalar_objective(mask, include_up):
    from clothdojo.data import retarget_dynamics as dynamics

    rng = np.random.default_rng(3)
    retargeter = dynamics.RobotDynamicsRetargeter.__new__(dynamics.RobotDynamicsRetargeter)
    retargeter.attach_body_indices = (0, 2)
    retargeter.profile = SimpleNamespace(attach_frames=[
        SimpleNamespace(position=(.01, .02, .03), rotation_xyzw=(0., 0., 0., 1.)),
        SimpleNamespace(position=(.03, -.02, .01), rotation_xyzw=(0., 0., 0., 1.)),
    ])
    retargeter._openness = lambda values: values[..., :2]
    q = rng.random((17, 4))
    bodies = rng.normal(size=(17, 3, 7)).astype(np.float32)
    bodies[..., 3:] /= np.linalg.norm(bodies[..., 3:], axis=-1, keepdims=True)
    target_pos = rng.normal(size=(2, 3))
    target_z = np.array([[0., 0., 1.]] * 2)
    target_y = np.array([[0., 1., 0.]] * 2) if include_up else None
    target_open = np.array([.2, .8])
    batch = retargeter._residual_batch(q, bodies, target_open, target_pos, target_z, mask, target_y)
    expected = []
    for coordinates, poses in zip(q, bodies, strict=True):
        frame = poses[None]
        expected.append(dynamics._task_space_residual(
            dynamics._attach_positions(frame, retargeter.profile.attach_frames, (0, 2))[0],
            coordinates[:2],
            dynamics._attach_approach_directions(
                frame, retargeter.profile.attach_frames, (0, 2),
            )[0],
            target_pos, target_open, target_z, mask,
            up_directions=dynamics._attach_axis_directions(
                frame, retargeter.profile.attach_frames, (0, 2), (0., 1., 0.),
            )[0], desired_up_directions=target_y,
        ))
    np.testing.assert_array_equal(batch, expected)


def test_rigid_graph_updates_commands_on_every_step(monkeypatch):
    import warp as wp

    runtime = ShirtFlattenRuntime.__new__(ShirtFlattenRuntime)
    runtime.model = SimpleNamespace(world_count=1, device=wp.get_device("cpu"))
    runtime._arm_action_count = 2
    runtime._graph_arm_target = wp.zeros((1, 2), device="cpu")
    runtime._graph_gripper_open = wp.zeros((1, 2), device="cpu")
    runtime._rigid_step_graph = object()
    assert runtime.rigid_cuda_graph_enabled
    commands = []

    def launch(graph):
        assert graph is runtime._rigid_step_graph
        commands.append((
            runtime._graph_arm_target.numpy().copy(), runtime._graph_gripper_open.numpy().copy(),
        ))

    monkeypatch.setattr(wp, "capture_launch", launch)
    for value in (0.25, 0.75):
        runtime.step_rigid(
            wp.full((1, 2), value, device="cpu"),
            wp.full((1, 2), 1 - value, device="cpu"),
        )
    np.testing.assert_array_equal(commands[0][0], [[0.25, 0.25]])
    np.testing.assert_array_equal(commands[1][0], [[0.75, 0.75]])
    np.testing.assert_array_equal(commands[1][1], [[0.25, 0.25]])


@pytest.mark.parametrize("substeps", [3, 6])
def test_eager_rigid_step_preserves_substep_count_and_buffer_parity(substeps):
    runtime = ShirtFlattenRuntime.__new__(ShirtFlattenRuntime)
    runtime.spec = SimpleNamespace(control_fps=50, sim_substeps=substeps)
    first, second = object(), object()
    runtime.state_0, runtime.state_1 = first, second
    calls = []
    runtime.apply_action = lambda arm, grip: calls.append((arm, grip))
    runtime._step_rigid = lambda dt: calls.append(dt)
    runtime._step_rigid_eager("arm", "grip")
    assert calls[0] == ("arm", "grip")
    np.testing.assert_allclose(calls[1:], [0.02 / substeps] * substeps)
    assert runtime.state_0 is (second if substeps % 2 else first)


def test_particle_free_scene_never_loads_or_initializes_cloth(monkeypatch):
    import newton

    from clothdojo.sim import scene as scene_module

    def forbidden(*args, **kwargs):
        pytest.fail("rigid-only retarget must not load or initialize cloth")

    monkeypatch.setattr(scene_module, "GarmentLibrary", forbidden)
    monkeypatch.setattr(scene_module, "load_garment_mesh", forbidden)
    scene = scene_module.build_scene(
        Path(__file__).resolve().parents[1] / "assets", robot_id="aloha2", device="cpu",
        include_cloth=False, robot_rigid_collision=False, robot_cloth_collision=False,
        spec=scene_module.ShirtFlattenSpec(garment_id="not-a-garment"),
    )
    assert scene.model.particle_count == 0
    assert scene.garment_particles_per_world == (0, 0)
    monkeypatch.setattr(newton.solvers, "SolverStyle3D", forbidden)
    monkeypatch.setattr("clothdojo.sim.runtime.ParticleGrasp", forbidden)
    monkeypatch.setattr(newton.solvers, "SolverMuJoCo", lambda *a, **kw: SimpleNamespace(
        mjw_model=None, reset=lambda *a, **kw: None,
    ))
    runtime = ShirtFlattenRuntime(scene, use_cuda_graph=False)
    assert runtime.cloth is runtime.grasp is runtime.collision is runtime.contacts is None
    assert runtime.rigid_state()["joint_q"].shape == (len(scene.robot_profile.home_joint_q),)


def test_partitions_balance_action_counts_and_keep_every_task(monkeypatch):
    sizes = [9, 8, 7, 6, 0]
    tasks = [(i, Path(str(i)), Path("out") / str(i)) for i in range(len(sizes))]
    monkeypatch.setattr(module, "TrajectoryReader", lambda path: SimpleNamespace(
        manifest=SimpleNamespace(action_count=sizes[int(path.name)]),
    ))
    partitions = module._retarget_partitions(tasks, 2)
    assert [[task[0] for task in group] for group in partitions] == [[0, 3, 4], [1, 2]]
    assert sorted(task for group in partitions for task in group) == tasks
    assert len(module._retarget_partitions(tasks, 10)) == len(tasks)


def test_worker_reuses_engines_across_garments_but_not_timings(monkeypatch, tmp_path):
    specs = [("a", 6), ("b", 10), ("c", 6)]
    tasks = [(i, Path(str(i)), tmp_path / str(i)) for i in range(3)]
    monkeypatch.setattr(module, "TrajectoryReader", lambda path: SimpleNamespace(
        manifest=SimpleNamespace(
            control_fps=50, sim_substeps=specs[int(path.name)][1],
            metadata={"asset_id": specs[int(path.name)][0]},
            scene=SimpleNamespace(task_name="shirt_flatten"),
        ),
    ))
    monkeypatch.setattr(
        module, "robot_profile", lambda *a, **kw: SimpleNamespace(asset="robot.usda"),
    )
    ik_calls, dynamics_calls, seen = [], [], []
    monkeypatch.setattr(
        module, "_RobotKinematics", lambda *a, **kw: ik_calls.append(kw) or object(),
    )

    def engine(*args, **kw):
        dynamics_calls.append(kw)
        return SimpleNamespace(**kw)

    def trajectory(source, destination, *, kinematics, dynamics, **kw):
        seen.append((int(source.name), kinematics, dynamics))
        return {"source": str(source)}

    monkeypatch.setattr(module, "RobotDynamicsRetargeter", engine)
    monkeypatch.setattr(module, "_retarget_trajectory", trajectory)
    result = module._retarget_partition(
        tasks, assets_root=tmp_path, target_robot="fr3_duo", target_asset_metadata={},
        position_tolerance_m=0.005, device="cpu",
    )
    assert len(ik_calls) == 1
    assert [call["sim_substeps"] for call in dynamics_calls] == [6, 10]
    assert [index for index, _ in result] == [0, 2, 1]
    assert seen[0][1] is seen[1][1] is seen[2][1]
    assert seen[0][2] is seen[1][2]
    assert seen[0][2] is not seen[2][2]


@pytest.mark.parametrize("devices,workers,assigned", [
    (("cuda:2", "cuda:5"), 1, ["cuda:2", "cuda:5"]),
    (("cuda:2",), 4, ["cuda:2", "cuda:2", "cuda:2"]),
    (("cuda:2", "cuda:5"), 2, ["cuda:2", "cuda:5", "cuda:2"]),
])
def test_parallel_collection_spawns_gpu_workers_and_publishes_source_order(
    monkeypatch, tmp_path, devices, workers, assigned,
):
    root, output = tmp_path / "source", tmp_path / "result"
    sources = [root / "trajectories" / str(i) for i in range(3)]
    monkeypatch.setattr(module, "_source_trajectories", lambda source: (sources, True))
    monkeypatch.setattr(module, "_validate_target_asset", lambda *a: {})
    monkeypatch.setattr(module, "robot_profile", lambda *a, **kw: SimpleNamespace(
        to_json_dict=lambda: {"robot_id": "fr3_duo"},
    ))
    monkeypatch.setattr(module, "TrajectoryReader", lambda path: SimpleNamespace(
        manifest=SimpleNamespace(action_count=10 - int(path.name)),
    ))
    calls = []

    class Pool:
        def __init__(self, *, max_workers, mp_context):
            assert max_workers == len(assigned)
            assert mp_context.get_start_method() == "spawn"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, tasks, **kw):
            assert function is module._retarget_partition
            calls.append((kw["device"], tasks))
            future = Future()
            future.set_result([(index, {
                "action_count": 10 - index,
                "metadata": {"asset_id": "shirt", "retarget": {
                    "source_trajectory": str(source), "validation": {
                        "training_eligible": False, "kinematic_valid": True,
                        "dynamics_valid": False,
                    },
                }},
            }) for index, source, _destination in tasks])
            return future

    monkeypatch.setattr(module, "ProcessPoolExecutor", Pool)
    result = module.retarget(
        root, output, target_robot="fr3_duo", devices=devices, workers_per_device=workers,
    )
    import json
    meta = json.loads((output / "meta.json").read_text())
    assert [entry["path"] for entry in meta["trajectories"]] == [
        f"trajectories/{i}" for i in range(3)
    ]
    assert [device for device, _tasks in calls] == assigned
    assert len({destination for _device, tasks in calls for _, _, destination in tasks}) == 3
    assert result["trajectory_count"] == 3
    assert not meta["training_eligible"]
    assert not list(tmp_path.glob(".result.tmp-*"))


@pytest.mark.parametrize("devices", [(), ("cuda:0", "cuda:0"), ("cuda", "cuda:1")])
def test_invalid_worker_devices_rejected_before_creating_output(tmp_path, devices):
    with pytest.raises(ValueError, match="devices"):
        module.retarget(tmp_path / "missing", tmp_path / "result", devices=devices)
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("workers", [0, -1, 1.5, True, "2"])
def test_invalid_workers_per_device_rejected_before_output(tmp_path, workers):
    with pytest.raises(ValueError, match="workers_per_device"):
        module.retarget(tmp_path / "missing", tmp_path / "result", workers_per_device=workers)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("tolerance", [0, -1, np.nan, np.inf])
def test_invalid_position_tolerance_rejected_before_output(tmp_path, tolerance):
    with pytest.raises(ValueError, match="positive and finite"):
        module.retarget(tmp_path / "missing", tmp_path / "result", position_tolerance_m=tolerance)
    assert not list(tmp_path.iterdir())


def test_empty_trajectory_rejected_before_gpu_or_output(monkeypatch, tmp_path):
    source, output = tmp_path / "source", tmp_path / "result"
    sources = [source / "trajectories" / str(i) for i in range(3)]
    monkeypatch.setattr(module, "_source_trajectories", lambda _: (sources, True))
    monkeypatch.setattr(module, "TrajectoryReader", lambda path: SimpleNamespace(
        manifest=SimpleNamespace(action_count=0 if path.name == "1" else 10),
    ))
    monkeypatch.setattr(module, "_validate_target_asset", lambda *a: pytest.fail("GPU setup"))
    with pytest.raises(ValueError, match="empty trajectories:") as error:
        module.retarget(source, output, workers_per_device=2)
    assert str(sources[1]) in str(error.value)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_same_device_parallelism_requires_explicit_gpu(tmp_path, device):
    with pytest.raises(ValueError, match="explicit cuda:N"):
        module.retarget(
            tmp_path / "missing", tmp_path / "result", device=device, workers_per_device=2,
        )
    assert not list(tmp_path.iterdir())


def test_parallel_worker_failure_does_not_publish_collection(monkeypatch, tmp_path):
    source, output = tmp_path / "source", tmp_path / "result"
    sources = [source / "trajectories" / str(i) for i in range(2)]
    monkeypatch.setattr(module, "_source_trajectories", lambda _: (sources, True))
    monkeypatch.setattr(module, "_validate_target_asset", lambda *a: {})
    monkeypatch.setattr(module, "robot_profile", lambda *a, **kw: object())
    monkeypatch.setattr(module, "TrajectoryReader", lambda _: SimpleNamespace(
        manifest=SimpleNamespace(action_count=10),
    ))

    class Pool:
        def __init__(self, **kw):
            assert kw["max_workers"] == 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, tasks, **kw):
            for _, _, destination in tasks:
                destination.mkdir(parents=True)
                (destination / "incomplete").touch()
            future = Future()
            future.set_exception(RuntimeError("worker failed"))
            return future

    monkeypatch.setattr(module, "ProcessPoolExecutor", Pool)
    with pytest.raises(RuntimeError, match="worker failed"):
        module.retarget(source, output, workers_per_device=2)
    assert not output.exists()
    assert not list(tmp_path.glob(".result.tmp-*"))
