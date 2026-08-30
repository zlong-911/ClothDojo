"""Read-only dynamics probe of authored robot USDs through Newton 1.5.

CPU MuJoCo isolates body/actuator/constraint behavior, without contacts.
GPU mode optionally uses Newton collision detection as in ClothDojo runtime.
No effort scaling, gain tuning, gravity compensation, or source edits.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")
import newton
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "outputs/robot_dynamics_audit"
sys.path.insert(0, str(ROOT / "src"))
from clothdojo.robots import robot_profile  # noqa: E402
from clothdojo.sim.scene import (  # noqa: E402
    ShirtFlattenSpec,
    _add_robot,
    _add_table,
    _register_solver_attributes,
)


def run(args):
    HERE.mkdir(parents=True, exist_ok=True)
    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
    _register_solver_attributes(newton, builder)
    profile = robot_profile(args.robot)
    source = ROOT / "assets" / profile.asset
    _add_robot(newton, wp, builder, source, profile, load_visual_shapes=False)
    arm_q = np.array(profile.arm_coord_indices)
    grip_q = np.array(profile.gripper_coord_indices)
    closed = np.tile(profile.gripper_closed_positions, 2)
    arm_joints = [builder.joint_q_start.index(int(q)) for q in arm_q]
    assert len(arm_q) >= 12 and len(grip_q) >= 2
    arm_dof = np.array([builder.joint_qd_start[i] for i in arm_joints])
    if args.contacts:
        _add_table(newton, wp, builder, ShirtFlattenSpec())
    model = builder.finalize(device=args.device)
    home = model.joint_q.numpy().copy()
    state, next_state = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    control.joint_target_q.assign(home)
    cpu = args.device == "cpu"
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_cpu=cpu,
        use_mujoco_contacts=cpu,
        disable_contacts=not args.contacts,
        disable_sensors=True,
        solver="newton",
        integrator="implicitfast",
        cone="pyramidal",
        iterations=100,
        ls_iterations=50,
        impratio=1.0,
        njmax=2048,
        nconmax=2048,
    )
    collision = contacts = None
    initial_contact_pairs = []
    if not cpu:
        collision = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            soft_contact_max=0,
            rigid_contact_max=4096,
            include_static_kinematic_pairs=False,
            deterministic=True,
        )
        contacts = collision.contacts()
    dt = 1 / (50 * args.substeps)
    steps = round(8 / dt)
    q_trace, qd_trace, targets, timestamps, contact_counts = [], [], [], [], []
    print(f"START {args.robot} {args.device} contacts={args.contacts} dt={dt}", flush=True)
    for step in range(steps):
        t = (step // args.substeps) / 50  # same 50 Hz zero-order-hold controls as runtime
        target = home.copy()
        if 2 <= t < 4:
            target[arm_q[[0, len(arm_q) // 2]]] += 0.06 * np.sin(np.pi * (t - 2))
        if 4 <= t < 7:
            u = min(t - 4, 1) if t < 6 else 7 - t
            amount = u * u * (3 - 2 * u)
            target[grip_q] = home[grip_q] + amount * (closed - home[grip_q])
        control.joint_target_q.assign(target)
        state.clear_forces()
        next_state.clear_forces()
        if collision and args.contacts:
            collision.collide(state, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            if step == 0:
                from collections import Counter

                a = contacts.rigid_contact_shape0.numpy()[:count]
                b = contacts.rigid_contact_shape1.numpy()[:count]
                pairs = Counter(zip(a.tolist(), b.tolist(), strict=True))
                initial_contact_pairs = [
                    {"shape0": model.shape_label[i], "shape1": model.shape_label[j], "count": n}
                    for (i, j), n in pairs.most_common(12)
                ]
                print("INITIAL_CONTACTS", json.dumps(initial_contact_pairs), flush=True)
            if count > 2048:
                (HERE / f"{args.robot}_contact_failure.json").write_text(
                    json.dumps(
                        {
                            "time_s": t,
                            "count": count,
                            "capacity": 2048,
                            "initial_contact_pairs": initial_contact_pairs,
                        },
                        indent=2,
                    )
                )
                raise RuntimeError(
                    f"Contact capacity exceeded ({count}>2048); "
                    "do not accept a truncated-contact run"
                )
        solver.step(state, next_state, control, contacts, dt)
        state, next_state = next_state, state
        if step % args.substeps == 0:
            q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
                raise RuntimeError(f"Nonfinite state at {t}")
            q_trace.append(q.copy())
            qd_trace.append(qd.copy())
            targets.append(target.copy())
            timestamps.append(t)
            if contacts:
                contact_counts.append(int(contacts.rigid_contact_count.numpy()[0]))
    q_trace, qd_trace, targets, timestamps = map(
        np.asarray, (q_trace, qd_trace, targets, timestamps)
    )
    hold = (timestamps > 1) & (timestamps < 2)
    closing = (timestamps > 5) & (timestamps < 6)
    returned = timestamps > 7.5
    arms_error = q_trace[:, arm_q] - targets[:, arm_q]
    closed_error = q_trace[:, grip_q] - targets[:, grip_q]
    report = {
        "robot": args.robot,
        "source": str(source),
        "device": args.device,
        "contacts": args.contacts,
        "solver": "Newton SolverMuJoCo",
        "duration_s": 8,
        "dt": dt,
        "effort_scale": 1,
        "added_gravity_compensation": False,
        "newton_gravity": model.gravity.numpy().tolist(),
        "mujoco_gravity": solver.mj_model.opt.gravity.tolist(),
        "mujoco_gravcomp": solver.mj_model.body_gravcomp.tolist(),
        "mujoco_disableflags": int(solver.mj_model.opt.disableflags),
        "finite_states": True,
        "arm_q_indices": arm_q.tolist(),
        "gripper_q_indices": grip_q.tolist(),
        "home_q": home.tolist(),
        "gripper_closed_target": closed.tolist(),
        "home_hold_max_joint_error_rad": float(np.abs(arms_error[hold]).max()),
        "home_hold_max_joint_speed_rad_s": float(np.abs(qd_trace[hold][:, arm_dof]).max()),
        "motion_max_joint_error_rad": float(
            np.abs(arms_error[(timestamps >= 2) & (timestamps < 4)]).max()
        ),
        "return_max_joint_error_rad": float(np.abs(arms_error[returned]).max()),
        "gripper_closed_max_coordinate_error": float(np.abs(closed_error[closing]).max()),
        "gripper_return_max_coordinate_error": float(np.abs(closed_error[returned]).max()),
        "max_rigid_contacts": max(contact_counts, default=0),
        "initial_contact_pairs": initial_contact_pairs,
        "joint_parameters": {
            key: getattr(model, key).numpy().tolist()
            for key in (
                "joint_target_ke",
                "joint_target_kd",
                "joint_effort_limit",
                "joint_velocity_limit",
                "joint_limit_lower",
                "joint_limit_upper",
                "joint_armature",
                "joint_damping",
                "joint_friction",
            )
            if getattr(model, key, None) is not None
        },
        "joint_labels": list(model.joint_label),
    }
    mode = "contacts" if args.contacts else "isolated"
    suffix = f"{args.robot}_{args.device.replace(':', '')}_{mode}_{args.substeps}"
    np.savez_compressed(
        HERE / f"{suffix}.npz", times=timestamps, joint_q=q_trace, joint_qd=qd_trace, target=targets
    )
    (HERE / f"{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("joint_parameters", "joint_labels", "home_q")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("robot")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--contacts", action="store_true")
    parser.add_argument("--substeps", type=int, default=6)
    run(parser.parse_args())
