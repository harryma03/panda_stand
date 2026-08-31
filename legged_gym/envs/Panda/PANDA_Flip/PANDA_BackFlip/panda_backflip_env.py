"""Panda BackFlip environment.

This class adapts the GO2 task to Panda and defines a successful backflip as
one complete signed rotation followed by a stable feet landing.

Actor single frame (45D):
    0:3     base_ang_vel
    3:6     projected_gravity
    6:9     commands
    9:21    dof_pos
    21:33   dof_vel
    33:45   actions

Actor history:
    45 * 10 = 450

Critic single frame (48D):
    0:3     commands
    3:15    dof_pos
    15:27   dof_vel
    27:39   actions
    39:42   base_lin_vel
    42:45   base_ang_vel
    45:48   projected_gravity

Critic history:
    48 * 3 = 144

Low-level simulation, torque, curriculum-push, and latency behavior remain
inherited from Go2_BackFlip.
"""

import torch
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs.GO2_Flip.GO2_BackFlip.GO2_BackFlip_env import Go2_BackFlip


class PandaBackFlip(Go2_BackFlip):
    """Panda backflip with explicit rotation and landing state."""

    def _init_buffers(self):
        """Create inherited buffers and Panda flip-state buffers."""
        super()._init_buffers()
        self.flip_angle = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.landing_flip_angle = torch.zeros_like(self.flip_angle)
        self.landing_ang_vel_y = torch.zeros_like(self.flip_angle)
        self.has_landed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.flip_success = torch.zeros_like(self.has_landed)
        self.landing_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def reset_idx(self, env_ids):
        """Log true flip metrics, then reset inherited and Panda state."""
        if len(env_ids) == 0:
            return

        mean_flip_angle = torch.mean(self.landing_flip_angle[env_ids])
        mean_landing_ang_vel = torch.mean(
            torch.abs(self.landing_ang_vel_y[env_ids])
        )
        landing_rate = torch.mean(self.has_landed[env_ids].float())
        success_rate = torch.mean(self.flip_success[env_ids].float())

        super().reset_idx(env_ids)

        self.extras["episode"]["flip_angle"] = mean_flip_angle
        self.extras["episode"]["landing_ang_vel_y"] = mean_landing_ang_vel
        self.extras["episode"]["landing_rate"] = landing_rate
        self.extras["episode"]["flip_success_rate"] = success_rate

        self.flip_angle[env_ids] = 0.0
        self.landing_flip_angle[env_ids] = 0.0
        self.landing_ang_vel_y[env_ids] = 0.0
        self.has_landed[env_ids] = False
        self.flip_success[env_ids] = False
        self.landing_steps[env_ids] = 0

    def check_jump(self):
        """Track signed rotation, touchdown, and stable full-flip success."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact.clone()

        command_active = self.commands[:, 2] > 0.0
        rotation_active = command_active & ~self.has_landed
        self.flip_angle[rotation_active] += (
            self.base_ang_vel[rotation_active, 1] * self.dt
        )

        all_feet_air = torch.all(~self.contact_filt, dim=1)
        self.was_in_flight |= all_feet_air & command_active

        touchdown = (
            torch.any(self.contact_filt, dim=1)
            & self.was_in_flight
            & ~self.has_landed
        )
        if torch.any(touchdown):
            self.has_landed[touchdown] = True
            self.has_jumped[touchdown] = True
            self.landing_flip_angle[touchdown] = self.flip_angle[touchdown]
            self.landing_ang_vel_y[touchdown] = self.base_ang_vel[touchdown, 1]
            self.landing_poses[touchdown] = self.root_states[touchdown, :2]

        settling = self.has_landed & ~self.flip_success
        self.landing_steps[settling] += 1

        projected_gravity = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        enough_feet = torch.sum(contact, dim=1) >= 2
        rotation_ok = (
            (self.landing_flip_angle >= self.cfg.rewards.min_flip_angle)
            & (self.landing_flip_angle <= self.cfg.rewards.max_flip_angle)
        )
        upright = (
            projected_gravity[:, 2]
            <= self.cfg.rewards.landing_upright_gravity_z
        )
        rotation_stopped = (
            torch.abs(self.base_ang_vel[:, 1])
            <= self.cfg.rewards.landing_max_ang_vel
        )
        height_ok = (
            self.root_states[:, 2]
            >= self.cfg.rewards.landing_min_height
        )
        inside_settle_window = (
            self.landing_steps
            <= self.cfg.rewards.landing_settle_steps
        )

        self.flip_success |= (
            self.has_landed
            & rotation_ok
            & upright
            & rotation_stopped
            & height_ok
            & enough_feet
            & inside_settle_window
        )

        # In inherited flight rewards, has_jumped is used to stop rewarding
        # take-off after touchdown. Full task success is tracked separately.
        self.has_jumped.copy_(self.has_landed)

    def check_termination(self):
        """Use the same contact, height and timeout termination as Go2."""
        super().check_termination()

    def _get_noise_scale_vec(self, cfg):
        """Noise vector for 45-D Panda actor observation."""
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs,
            dtype=torch.float,
            device=self.device,
        )

        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales

        # 0:3 base angular velocity
        noise_vec[0:3] = (
            noise_scales.ang_vel
            * self.obs_scales.ang_vel
        )

        # 3:6 projected gravity
        noise_vec[3:6] = noise_scales.gravity

        # 6:9 commands
        noise_vec[6:9] = 0.0

        # 9:21 q
        noise_vec[9:21] = (
            noise_scales.dof_pos
            * self.obs_scales.dof_pos
        )

        # 21:33 dq
        noise_vec[21:33] = (
            noise_scales.dof_vel
            * self.obs_scales.dof_vel
        )

        # 33:45 actions
        noise_vec[33:45] = 0.0

        return noise_vec

    def update_obs_latency_buffer(self):
        """GO2 latency mechanism with projected gravity instead of Euler XYZ."""

        if self.cfg.domain_rand.randomize_obs_motor_latency:
            q = (
                self.dof_pos - self.default_dof_pos
            ) * self.obs_scales.dof_pos

            dq = (
                self.dof_vel
                * self.obs_scales.dof_vel
            )

            self.obs_motor_latency_buffer[:, :, 1:] = (
                self.obs_motor_latency_buffer[
                    :, :, :self.cfg.domain_rand.range_obs_motor_latency[1]
                ].clone()
            )

            self.obs_motor_latency_buffer[:, :, 0] = torch.cat(
                (q, dq),
                dim=1,
            ).clone()

        if self.cfg.domain_rand.randomize_obs_imu_latency:
            self.gym.refresh_actor_root_state_tensor(self.sim)

            self.base_quat[:] = self.root_states[:, 3:7]

            self.base_ang_vel[:] = quat_rotate_inverse(
                self.base_quat,
                self.root_states[:, 10:13],
            )

            projected_gravity = quat_rotate_inverse(
                self.base_quat,
                self.gravity_vec,
            )

            self.obs_imu_latency_buffer[:, :, 1:] = (
                self.obs_imu_latency_buffer[
                    :, :, :self.cfg.domain_rand.range_obs_imu_latency[1]
                ].clone()
            )

            self.obs_imu_latency_buffer[:, :, 0] = torch.cat(
                (
                    self.base_ang_vel * self.obs_scales.ang_vel,
                    projected_gravity,
                ),
                dim=1,
            ).clone()

    def compute_observations(self):
        """Build 45-D actor frames and 48-D privileged critic frames."""

        env_ids = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )

        q = (
            self.dof_pos - self.default_dof_pos
        ) * self.obs_scales.dof_pos

        dq = (
            self.dof_vel
            * self.obs_scales.dof_vel
        )

        current_gravity = quat_rotate_inverse(
            self.base_quat,
            self.gravity_vec,
        )

        # --------------------------------------------------------
        # Motor observation
        # --------------------------------------------------------
        if self.cfg.domain_rand.randomize_obs_motor_latency:
            obs_motor = self.obs_motor_latency_buffer[
                env_ids,
                :,
                self.obs_motor_latency_simstep.long(),
            ]
        else:
            obs_motor = torch.cat(
                (q, dq),
                dim=1,
            )

        # --------------------------------------------------------
        # IMU observation
        # --------------------------------------------------------
        if self.cfg.domain_rand.randomize_obs_imu_latency:
            obs_imu = self.obs_imu_latency_buffer[
                env_ids,
                :,
                self.obs_imu_latency_simstep.long(),
            ]
        else:
            obs_imu = torch.cat(
                (
                    self.base_ang_vel * self.obs_scales.ang_vel,
                    current_gravity,
                ),
                dim=1,
            )

        delayed_ang_vel = obs_imu[:, 0:3]
        delayed_gravity = obs_imu[:, 3:6]

        delayed_q = obs_motor[:, 0:12]
        delayed_dq = obs_motor[:, 12:24]

        # --------------------------------------------------------
        # Actor single frame = 45D
        # Same ordering as Panda Handstand
        # --------------------------------------------------------
        obs_single = torch.cat(
            (
                delayed_ang_vel,          # 0:3
                delayed_gravity,          # 3:6
                self.commands[:, :3],     # 6:9
                delayed_q,                # 9:21
                delayed_dq,               # 21:33
                self.actions,             # 33:45
            ),
            dim=-1,
        )

        if obs_single.shape[1] != 45:
            raise RuntimeError(
                "Panda BackFlip single actor observation must be 45-D, "
                f"got {tuple(obs_single.shape)}"
            )

        if self.add_noise:
            obs_now = (
                obs_single.clone()
                + (
                    2.0 * torch.rand_like(obs_single) - 1.0
                )
                * self.noise_scale_vec
                * self.cfg.noise.noise_level
            )
        else:
            obs_now = obs_single.clone()

        # --------------------------------------------------------
        # Critic single frame = 48D
        #
        # Keep GO2 privileged ordering as closely as possible:
        # remove only the original two dummy zeros and replace
        # Euler XYZ with projected gravity.
        # --------------------------------------------------------
        privileged_single = torch.cat(
            (
                self.commands[:, :3],                         # 3
                q,                                            # 12
                dq,                                           # 12
                self.actions,                                 # 12
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                current_gravity,                              # 3
            ),
            dim=-1,
        )

        if privileged_single.shape[1] != 48:
            raise RuntimeError(
                "Panda BackFlip single critic observation must be 48-D, "
                f"got {tuple(privileged_single.shape)}"
            )

        # --------------------------------------------------------
        # History
        # --------------------------------------------------------
        self.obs_history.append(obs_now)
        self.critic_history.append(privileged_single)

        obs_buf_all = torch.stack(
            [
                self.obs_history[i]
                for i in range(self.obs_history.maxlen)
            ],
            dim=1,
        )

        self.obs_buf = obs_buf_all.reshape(
            self.num_envs,
            -1,
        )

        self.privileged_obs_buf = torch.cat(
            [
                self.critic_history[i]
                for i in range(self.cfg.env.c_frame_stack)
            ],
            dim=1,
        )

        expected_actor = (
            self.cfg.env.frame_stack
            * self.cfg.env.num_single_obs
        )

        expected_critic = (
            self.cfg.env.c_frame_stack
            * self.cfg.env.single_num_privileged_obs
        )

        if self.obs_buf.shape[1] != expected_actor:
            raise RuntimeError(
                f"Panda BackFlip actor observation must be "
                f"{expected_actor}-D, got {tuple(self.obs_buf.shape)}"
            )

        if self.privileged_obs_buf.shape[1] != expected_critic:
            raise RuntimeError(
                f"Panda BackFlip critic observation must be "
                f"{expected_critic}-D, "
                f"got {tuple(self.privileged_obs_buf.shape)}"
            )

    def _rotation_ok(self):
        return (
            (self.landing_flip_angle >= self.cfg.rewards.min_flip_angle)
            & (self.landing_flip_angle <= self.cfg.rewards.max_flip_angle)
        )

    def _reward_before_setting(self):
        """Move to the configured crouch before the jump command."""
        pose_error = torch.sum(
            torch.abs(self.dof_pos - self.lie_joint_pos), dim=1
        )
        return (
            torch.exp(-pose_error / 4.0)
            * (self.commands[:, 2] == 0.0)
        )

    def _reward_angle_y(self):
        """Use Go2's dense positive pitch-rate reward."""
        rew = (
            self.base_ang_vel[:, 1]
            * (self.base_ang_vel[:, 1] > 0.0)
            * (self.commands[:, 2] == 1.0)
            * ~self.was_in_flight
        )
        rew += (
            3.0
            * self.base_ang_vel[:, 1]
            * (self.base_ang_vel[:, 1] > 0.0)
            * self.was_in_flight
            * ~self.has_jumped
        )
        return torch.clip(rew, max=20.0)

    def _reward_flip_progress(self):
        """Dense positive rotation reward, capped after one revolution."""
        active = (
            (self.commands[:, 2] > 0.0)
            & ~self.has_landed
            & (self.flip_angle < self.cfg.rewards.target_flip_angle)
        )
        positive_rate = torch.clamp(
            self.base_ang_vel[:, 1], min=0.0, max=10.0
        )
        return positive_rate * active

    def _reward_flip_completion(self):
        """Reward touchdown close to one complete revolution."""
        angle_error = torch.abs(
            self.landing_flip_angle - self.cfg.rewards.target_flip_angle
        )
        return (
            torch.exp(-2.0 * angle_error)
            * self.has_landed
        )

    def _reward_flip_success(self):
        return self.flip_success.float()

    def _reward_base_height_flight(self):
        height_error = (
            self.root_states[:, 2] - self.cfg.rewards.target_height
        )
        return (
            torch.exp(-torch.abs(height_error) * 5.0)
            * self.was_in_flight
            * ~self.has_landed
            * 6.0
        )

    def _reward_base_height_stance(self):
        height_error = (
            self.root_states[:, 2]
            - self.cfg.rewards.stance_height_target
        )
        return (
            torch.exp(-torch.abs(height_error) * 10.0)
            * self.has_landed
            * (self.max_ang_vel_y > 7.0)
        )

    def _reward_orientation(self):
        return (
            torch.exp(-torch.abs(self.base_euler_xyz).sum(dim=1))
            * self.has_landed
            * (self.max_ang_vel_y > 7.0)
        )

    def _reward_land_pos(self):
        land_error = self.init_state[:, :2] - self.root_states[:, :2]
        return (
            torch.exp(-torch.sum(torch.abs(land_error), dim=1))
            * self.has_landed
        )

    def _reward_line_vel_stance(self):
        return torch.sum(torch.abs(self.base_lin_vel), dim=1)
