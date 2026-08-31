"""Panda spring jump environment."""

import torch

from legged_gym.envs.GO2_Flip.GO2_Spring_Jump.GO2_Spring_Jump_env import (
    GO2_Spring_Jump_Robot,
)


class PandaSpringJump(GO2_Spring_Jump_Robot):
    """Panda-specific preload, takeoff, flight and landing task."""

    def _ensure_contact_counter(self):
        if not hasattr(self, "base_contact_counter"):
            self.base_contact_counter = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )

    def check_termination(self):
        self._ensure_contact_counter()
        base_force = torch.norm(
            self.contact_forces[:, self.termination_contact_indices, :], dim=-1
        ).amax(dim=1)
        contact = base_force > self.cfg.rewards.base_contact_force_threshold
        self.base_contact_counter = torch.where(
            contact, self.base_contact_counter + 1, torch.zeros_like(self.base_contact_counter)
        )
        self.reset_buf = (
            self.base_contact_counter >= self.cfg.rewards.base_contact_sustain_steps
        ) | (self.root_states[:, 2] <= self.cfg.env.reset_height)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.commands[env_ids, 0:2] = 0.0
        if hasattr(self, "base_contact_counter"):
            self.base_contact_counter[env_ids] = 0

    def _reward_before_setting(self):
        pose_error = torch.mean(torch.abs(self.dof_pos - self.lie_joint_pos), dim=1)
        return torch.exp(-4.0 * pose_error) * (self.commands[:, 2] == 0).float()

    def _reward_preload_height(self):
        error = torch.abs(self.root_states[:, 2] - self.cfg.rewards.preload_height_target)
        return torch.exp(-10.0 * error) * (self.commands[:, 2] == 0).float()

    def _reward_line_z(self):
        active = (self.commands[:, 2] == 1) & (~self.was_in_flight) & (~self.has_jumped)
        error = torch.square(
            self.base_lin_vel[:, 2] - self.cfg.rewards.takeoff_velocity_target
        )
        return torch.exp(-error / 0.25) * active.float()

    def _reward_base_height_flight(self):
        active = self.was_in_flight & (~self.has_jumped)
        error = torch.abs(self.root_states[:, 2] - self.cfg.rewards.flight_height_target)
        return torch.exp(-8.0 * error) * active.float()

    def _reward_base_height_stance(self):
        active = self.has_jumped
        error = torch.abs(self.root_states[:, 2] - self.cfg.rewards.base_height_target)
        return torch.exp(-10.0 * error) * active.float()

    def _reward_flight_leg_pose(self):
        active = self.was_in_flight & (~self.has_jumped)
        descending = self.base_lin_vel[:, 2] < -0.25
        target = torch.where(
            descending.unsqueeze(1), self.default_dof_pos, self.lie_joint_pos.unsqueeze(0)
        )
        pose_error = torch.mean(torch.abs(self.dof_pos - target), dim=1)
        return torch.exp(-4.0 * pose_error) * active.float()

    def _reward_land_pos(self):
        error = torch.norm(self.root_states[:, :2] - self.init_state[:, :2], dim=1)
        upright = torch.abs(self.base_euler_xyz).sum(dim=1) < 0.6
        return torch.exp(-3.0 * error) * self.has_jumped.float() * upright.float()

    def _reward_landing_stability(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stable = torch.exp(-torch.sum(torch.square(self.base_lin_vel), dim=1) / 0.5)
        upright = torch.exp(-torch.abs(self.base_euler_xyz).sum(dim=1) * 4.0)
        return stable * upright * contact.all(dim=1).float() * self.has_jumped.float()
