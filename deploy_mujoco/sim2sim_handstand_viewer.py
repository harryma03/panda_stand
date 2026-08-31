#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Panda Handstand Sim2Sim: Isaac Gym policy -> ONNXRuntime -> MuJoCo

Default:
  - Panda MJCF: resources/robots/panda/panda3v2.xml
  - ONNX: logs/panda_handstand/exported/policies/policy_1.onnx
  - obs: 48
  - action: 12
  - MuJoCo physics: 200 Hz (dt=0.005)
  - policy: 50 Hz (decimation=4)
  - first 300 physics steps (1.5 s): hold default quadruped pose with PD
  - command input disabled by default

Before running:
  1) Ensure panda3v2.xml contains a floor geom named "floor".
  2) pip install onnxruntime mujoco scipy
"""

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

from legged_gym import LEGGED_GYM_ROOT_DIR


# ============================================================
# Panda Sim2Sim configuration
# ============================================================

class Sim2simCfg:
    class env:
        num_actions = 12
        frame_stack = 1
        num_single_obs = 48
        num_observations = 48

    class control:
        action_scale = 0.25
        decimation = 4

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05

        clip_observations = 100.0
        clip_actions = 100.0

    class sim_config:
        mujoco_model_path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "resources",
            "robots",
            "panda",
            "panda3v2.xml",
        )
        sim_duration = 120.0
        dt = 0.005                 # 200 Hz physics
        decimation = 4             # 50 Hz policy
        warmup_steps = 300         # 1.5 s
        initial_base_z = 0.558     # use the XML's original reset height

    class robot_config:
        # Joint/action order:
        # FL hip thigh calf,
        # FR hip thigh calf,
        # RL hip thigh calf,
        # RR hip thigh calf

        kps = np.array([
            45.0, 120.0, 160.0,   # FL
            45.0, 120.0, 160.0,   # FR
            45.0, 120.0, 160.0,   # RL
            45.0, 120.0, 160.0,   # RR
        ], dtype=np.float64)

        kds = np.array([
            2.0, 4.5, 8.0,        # FL
            2.0, 4.5, 8.0,        # FR
            2.0, 4.5, 8.0,        # RL
            2.0, 4.5, 8.0,        # RR
        ], dtype=np.float64)

        default_dof_pos = np.array([
            0.0, 0.52, -1.05,     # FL
            0.0, 0.52, -1.05,     # FR
            0.0, 0.52, -1.05,     # RL
            0.0, 0.52, -1.05,     # RR
        ], dtype=np.float64)

        # Match the Panda MJCF motor ctrlrange:
        # hip/abduction ±60, thigh ±60, calf/knee ±90 Nm
        tau_limit = np.array([
            60.0, 60.0, 90.0,     # FL
            60.0, 60.0, 90.0,     # FR
            60.0, 60.0, 90.0,     # RL
            60.0, 60.0, 90.0,     # RR
        ], dtype=np.float64)


EXPECTED_JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

EXPECTED_ACTUATORS = [
    "FL_hip_joint_ctrl", "FL_thigh_joint_ctrl", "FL_calf_joint_ctrl",
    "FR_hip_joint_ctrl", "FR_thigh_joint_ctrl", "FR_calf_joint_ctrl",
    "RL_hip_joint_ctrl", "RL_thigh_joint_ctrl", "RL_calf_joint_ctrl",
    "RR_hip_joint_ctrl", "RR_thigh_joint_ctrl", "RR_calf_joint_ctrl",
]


# ============================================================
# ONNX policy
# ============================================================

class OnnxPolicy:
    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX policy not found: {model_path}")

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]

        print("\n========== ONNX ==========")
        print("model :", os.path.abspath(model_path))
        print("input :", self.input.name, self.input.shape, self.input.type)
        print("output:", self.output.name, self.output.shape, self.output.type)

        # This Panda policy is expected to be 48 -> 12.
        if len(self.input.shape) != 2 or self.input.shape[-1] != 48:
            raise RuntimeError(
                f"Expected ONNX input [batch, 48], got {self.input.shape}"
            )
        if len(self.output.shape) != 2 or self.output.shape[-1] != 12:
            raise RuntimeError(
                f"Expected ONNX output [batch, 12], got {self.output.shape}"
            )

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        out = self.session.run(
            [self.output.name],
            {self.input.name: obs},
        )[0]
        return np.asarray(out, dtype=np.float32)


# ============================================================
# MuJoCo helpers
# ============================================================

def validate_mujoco_model(model: mujoco.MjModel):
    print("\n========== MUJOCO MODEL ==========")
    print("nq =", model.nq)
    print("nv =", model.nv)
    print("nu =", model.nu)
    print("njnt =", model.njnt)
    print("nbody =", model.nbody)

    if model.nq != 19:
        raise RuntimeError(f"Expected nq=19 (free base + 12 joints), got {model.nq}")
    if model.nv != 18:
        raise RuntimeError(f"Expected nv=18 (6 base + 12 joint velocities), got {model.nv}")
    if model.nu != 12:
        raise RuntimeError(f"Expected nu=12 actuators, got {model.nu}")

    # Check floor.
    floor_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "floor",
    )
    if floor_id < 0:
        raise RuntimeError(
            "\nPanda XML does not contain a geom named 'floor'.\n"
            "Add the floor geom to <worldbody> before running Sim2Sim."
        )

    # Collect hinge-joint names, skipping the floating base.
    joint_names = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            j,
        )
        if name != "floating_base":
            joint_names.append(name)

    actuator_names = [
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            i,
        )
        for i in range(model.nu)
    ]

    print("\nJoint order:")
    for i, name in enumerate(joint_names):
        print(f"  {i:2d}: {name}")

    print("\nActuator order:")
    for i, name in enumerate(actuator_names):
        print(f"  {i:2d}: {name}")

    if joint_names != EXPECTED_JOINTS:
        raise RuntimeError(
            "\nMuJoCo joint order does not match the Panda policy order.\n"
            f"Expected: {EXPECTED_JOINTS}\n"
            f"Got     : {joint_names}"
        )

    if actuator_names != EXPECTED_ACTUATORS:
        raise RuntimeError(
            "\nMuJoCo actuator order does not match the Panda policy order.\n"
            f"Expected: {EXPECTED_ACTUATORS}\n"
            f"Got     : {actuator_names}"
        )

    print("\nPanda joint/actuator order: OK")
    print("Floor: OK")


def get_state(data: mujoco.MjData):
    """
    Panda MJCF:
      qpos[0:3]   = base xyz
      qpos[3:7]   = base quaternion [w, x, y, z]
      qpos[7:19]  = 12 joint positions

      qvel[0:3]   = base linear velocity in world frame
      qvel[3:6]   = free-joint angular velocity in local body frame
      qvel[6:18]  = 12 joint velocities
    """
    q = data.qpos[7:19].copy().astype(np.float64)
    dq = data.qvel[6:18].copy().astype(np.float64)

    # MuJoCo quaternion [w, x, y, z] -> SciPy [x, y, z, w]
    quat_xyzw = data.qpos[3:7].copy().astype(np.float64)[[1, 2, 3, 0]]
    rot = R.from_quat(quat_xyzw)

    # Base linear velocity: world -> body frame.
    base_lin_vel_body = rot.apply(
        data.qvel[0:3],
        inverse=True,
    ).astype(np.float64)

    # MuJoCo free-joint rotational qvel is already local/body-frame.
    base_ang_vel_body = data.qvel[3:6].copy().astype(np.float64)

    # World gravity direction expressed in body frame.
    projected_gravity = rot.apply(
        np.array([0.0, 0.0, -1.0], dtype=np.float64),
        inverse=True,
    ).astype(np.float64)

    base_pos = data.qpos[0:3].copy().astype(np.float64)

    return (
        q,
        dq,
        base_lin_vel_body,
        base_ang_vel_body,
        projected_gravity,
        base_pos,
    )


def build_observation(
    data: mujoco.MjData,
    cfg,
    last_action: np.ndarray,
    command: np.ndarray,
):
    (
        q,
        dq,
        _base_lin_vel,
        omega,
        gvec,
        base_pos,
    ) = get_state(data)

    obs = np.zeros((1, cfg.env.num_single_obs), dtype=np.float32)

    # IMPORTANT:
    # The current handstand policy uses three zeros in obs[0:3],
    # matching the Go2-handstand-style 48-D interface.
    obs[0, 0:3] = 0.0

    # body-frame angular velocity
    obs[0, 3:6] = (
        omega * cfg.normalization.obs_scales.ang_vel
    ).astype(np.float32)

    # projected gravity
    obs[0, 6:9] = gvec.astype(np.float32)

    # command: vx, vy, yaw_rate
    obs[0, 9] = (
        command[0] * cfg.normalization.obs_scales.lin_vel
    )
    obs[0, 10] = (
        command[1] * cfg.normalization.obs_scales.lin_vel
    )
    obs[0, 11] = (
        command[2] * cfg.normalization.obs_scales.ang_vel
    )

    # joint position error relative to training default pose
    obs[0, 12:24] = (
        (q - cfg.robot_config.default_dof_pos)
        * cfg.normalization.obs_scales.dof_pos
    ).astype(np.float32)

    # joint velocity
    obs[0, 24:36] = (
        dq * cfg.normalization.obs_scales.dof_vel
    ).astype(np.float32)

    # previous policy action
    obs[0, 36:48] = last_action.astype(np.float32)

    obs = np.clip(
        obs,
        -cfg.normalization.clip_observations,
        cfg.normalization.clip_observations,
    )

    return obs, q, dq, base_pos


def pd_control(
    target_q_offset: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
    cfg,
):
    """
    Training-equivalent position target:
        q_des = default_dof_pos + action_scale * action

    target_q_offset is action_scale * action.
    """
    q_des = cfg.robot_config.default_dof_pos + target_q_offset
    target_dq = np.zeros(cfg.env.num_actions, dtype=np.float64)

    tau = (q_des - q) * kp + (target_dq - dq) * kd
    return tau


# ============================================================
# Main simulation
# ============================================================

def run_mujoco(
    policy: OnnxPolicy,
    cfg,
    model_path: str,
    pd_only: bool = False,
    enable_cmd: bool = False,
    warmup_steps: int = 300,
):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MuJoCo XML not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = cfg.sim_config.dt

    validate_mujoco_model(model)

    data = mujoco.MjData(model)

    # Explicit reset.
    data.qpos[0:3] = np.array(
        [0.0, 0.0, cfg.sim_config.initial_base_z],
        dtype=np.float64,
    )
    # MuJoCo quaternion is [w, x, y, z].
    data.qpos[3:7] = np.array(
        [1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    data.qpos[7:19] = cfg.robot_config.default_dof_pos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    mujoco.mj_forward(model, data)

    # Optional command/arrow input.
    vel_arrow = None
    if enable_cmd:
        try:
            from viewer_utils import VelocityArrowViewer
        except ImportError as exc:
            raise ImportError(
                "enable_cmd=True but viewer_utils.py cannot be imported."
            ) from exc

        # Keep the original viewer limits.
        vel_arrow = VelocityArrowViewer(
            max_cmd=(1.5, 1.0, 1.5)
        )

    action = np.zeros(cfg.env.num_actions, dtype=np.float32)
    target_q_offset = np.zeros(cfg.env.num_actions, dtype=np.float64)
    command = np.zeros(3, dtype=np.float64)

    total_steps = int(
        cfg.sim_config.sim_duration / cfg.sim_config.dt
    )

    print("\n========== CONTROL ==========")
    print(f"physics dt       : {cfg.sim_config.dt:.6f} s")
    print(f"physics rate     : {1.0 / cfg.sim_config.dt:.1f} Hz")
    print(f"decimation       : {cfg.sim_config.decimation}")
    print(
        "policy rate      : "
        f"{1.0 / (cfg.sim_config.dt * cfg.sim_config.decimation):.1f} Hz"
    )
    print(f"warmup steps     : {warmup_steps}")
    print(
        f"warmup duration  : {warmup_steps * cfg.sim_config.dt:.3f} s"
    )
    print(f"PD-only          : {pd_only}")
    print(f"command input    : {enable_cmd}")
    print("=============================\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25
        viewer.cam.lookat[:] = np.array(
            [0.0, 0.0, 0.65],
            dtype=np.float64,
        )

        step_idx = 0

        while viewer.is_running() and step_idx < total_steps:
            step_start = time.time()

            # ----------------------------------------------------
            # Command
            # ----------------------------------------------------
            if enable_cmd:
                cmd = vel_arrow.update_cmd()
                command[:] = np.asarray(cmd[:3], dtype=np.float64)
            else:
                # First Sim2Sim test: keep command exactly zero.
                command[:] = 0.0

            # ----------------------------------------------------
            # State
            # ----------------------------------------------------
            (
                q,
                dq,
                _base_lin_vel,
                _omega,
                _gvec,
                base_pos,
            ) = get_state(data)

            # ----------------------------------------------------
            # Warmup: hold quadruped default pose.
            #
            # During warmup action is also kept at zero, so the
            # "previous action" observation is consistent when the
            # policy takes control.
            # ----------------------------------------------------
            if pd_only or step_idx < warmup_steps:
                action[:] = 0.0
                target_q_offset[:] = 0.0

            else:
                # 200 Hz physics -> 50 Hz policy
                if step_idx % cfg.sim_config.decimation == 0:
                    obs, q, dq, base_pos = build_observation(
                        data=data,
                        cfg=cfg,
                        last_action=action,
                        command=command,
                    )

                    policy_action = policy(obs)[0]

                    action[:] = np.clip(
                        policy_action,
                        -cfg.normalization.clip_actions,
                        cfg.normalization.clip_actions,
                    )

                    target_q_offset[:] = (
                        action.astype(np.float64)
                        * cfg.control.action_scale
                    )

            # ----------------------------------------------------
            # PD -> torque
            # ----------------------------------------------------
            tau = pd_control(
                target_q_offset=target_q_offset,
                q=q,
                kp=cfg.robot_config.kps,
                dq=dq,
                kd=cfg.robot_config.kds,
                cfg=cfg,
            )

            tau = np.clip(
                tau,
                -cfg.robot_config.tau_limit,
                cfg.robot_config.tau_limit,
            )

            data.ctrl[:] = tau

            # ----------------------------------------------------
            # Physics
            # ----------------------------------------------------
            mujoco.mj_step(model, data)

            if enable_cmd:
                vel_arrow.draw_arrows(viewer, data)

            viewer.sync()

            # Debug every 2.5 s.
            if step_idx % 500 == 0:
                print(
                    f"step={step_idx:6d}  "
                    f"t={step_idx * cfg.sim_config.dt:7.3f}s  "
                    f"base_z={data.qpos[2]:.4f}  "
                    f"|tau|max={np.max(np.abs(tau)):.2f}  "
                    f"|action|max={np.max(np.abs(action)):.3f}"
                )

            # Real-time playback.
            sleep_time = (
                model.opt.timestep - (time.time() - step_start)
            )
            if sleep_time > 0:
                time.sleep(sleep_time)

            step_idx += 1


def main():
    default_model = Sim2simCfg.sim_config.mujoco_model_path

    default_policy = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        "panda_handstand",
        "exported",
        "policies",
        "policy_1.onnx",
    )

    parser = argparse.ArgumentParser(
        description="Panda Handstand ONNX MuJoCo Sim2Sim"
    )

    parser.add_argument(
        "--load_model",
        type=str,
        default=default_policy,
        help="Path to policy_1.onnx",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="Path to panda3v2.xml",
    )

    parser.add_argument(
        "--pd_only",
        action="store_true",
        help="Only hold the default quadruped pose; do not run the RL policy.",
    )

    parser.add_argument(
        "--enable_cmd",
        action="store_true",
        help="Enable VelocityArrowViewer command input.",
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=Sim2simCfg.sim_config.warmup_steps,
        help="PD-only warmup physics steps before policy takeover.",
    )

    args = parser.parse_args()

    print("MuJoCo XML :", os.path.abspath(args.model))
    print("ONNX policy:", os.path.abspath(args.load_model))

    policy = OnnxPolicy(args.load_model)

    run_mujoco(
        policy=policy,
        cfg=Sim2simCfg(),
        model_path=args.model,
        pd_only=args.pd_only,
        enable_cmd=args.enable_cmd,
        warmup_steps=args.warmup_steps,
    )


if __name__ == "__main__":
    main()
