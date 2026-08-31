"""Panda command-conditioned periodic jump environment."""

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply

from legged_gym.envs.Go2_MoB.GO2_JUMP.go2_jump_env import GO2_JUMP_Robot


class PandaJump(GO2_JUMP_Robot):
    """One-shot crouch, forward high jump and stable four-foot landing."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.base_contact_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.continuous_air_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.jump_was_airborne = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.jump_has_landed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.jump_has_touched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.jump_front_supported = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.touchdown_front_first = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.touchdown_rear_first = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.touchdown_simultaneous = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.jump_max_height = self.root_states[:, 2].clone()
        self.jump_max_air_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.jump_assisted = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.roll_assisted = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Cumulative world-frame roll angle while airborne; the side-flip
        # progress and completion rewards are built on it.
        self.cum_flip_roll = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        # Whether each robot is airborne *right now*.  Landing/flip rewards
        # must use this instead of the historical jump_was_airborne flag:
        # a belly-up robot satisfies "was airborne, never landed" forever
        # and harvested landing_foot_level/landing_pose without jumping.
        self.airborne_now = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Keep privileged observations meaningful when randomization is disabled.
        self.env_frictions.fill_(self.cfg.terrain.static_friction)
        self.body_mass.fill_(self.cfg.rewards.robot_mass)
        self.flight_tuck_dof_pos = self.default_dof_pos.clone()
        self.crouch_dof_pos = self.default_dof_pos.clone()
        self.takeoff_dof_pos = self.default_dof_pos.clone()
        self.landing_dof_pos = self.default_dof_pos.clone()
        self.landing_absorb_dof_pos = self.default_dof_pos.clone()
        self.front_support_dof_pos = self.default_dof_pos.clone()
        for index, name in enumerate(self.dof_names):
            is_front = name.startswith("FL_") or name.startswith("FR_")
            if "thigh" in name:
                self.flight_tuck_dof_pos[:, index] = (
                    self.cfg.rewards.flight_front_thigh
                    if is_front
                    else self.cfg.rewards.flight_rear_thigh
                )
                self.crouch_dof_pos[:, index] = self.cfg.rewards.crouch_thigh_target
                self.takeoff_dof_pos[:, index] = self.cfg.rewards.takeoff_thigh_target
                self.landing_dof_pos[:, index] = (
                    self.cfg.rewards.landing_front_thigh
                    if is_front
                    else self.cfg.rewards.landing_rear_thigh
                )
                self.landing_absorb_dof_pos[:, index] = (
                    self.cfg.rewards.landing_absorb_thigh
                )
                self.front_support_dof_pos[:, index] = (
                    self.cfg.rewards.front_support_thigh
                    if is_front
                    else self.cfg.rewards.rear_follow_thigh
                )
            elif "calf" in name:
                self.flight_tuck_dof_pos[:, index] = (
                    self.cfg.rewards.flight_front_calf
                    if is_front
                    else self.cfg.rewards.flight_rear_calf
                )
                self.crouch_dof_pos[:, index] = self.cfg.rewards.crouch_calf_target
                self.takeoff_dof_pos[:, index] = self.cfg.rewards.takeoff_calf_target
                self.landing_dof_pos[:, index] = (
                    self.cfg.rewards.landing_front_calf
                    if is_front
                    else self.cfg.rewards.landing_rear_calf
                )
                self.landing_absorb_dof_pos[:, index] = (
                    self.cfg.rewards.landing_absorb_calf
                )
                self.front_support_dof_pos[:, index] = (
                    self.cfg.rewards.front_support_calf
                    if is_front
                    else self.cfg.rewards.rear_follow_calf
                )
        self._set_forward_command()

    def _set_forward_command(self, env_ids=None):
        if env_ids is None:
            self.commands[:, 0] = self.cfg.rewards.forward_velocity_target
            self.commands[:, 1:3] = 0.0
        else:
            self.commands[env_ids, 0] = self.cfg.rewards.forward_velocity_target
            self.commands[env_ids, 1:3] = 0.0

    def step(self, actions):
        # play.py writes zero commands once; this task must always jump forward.
        self._set_forward_command()
        return super().step(actions)

    def _resample_commands(self, env_ids):
        self._set_forward_command(env_ids)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._set_forward_command()
        phase = self._get_phase()
        assist_start = self.cfg.rewards.flight_start_phase - 0.025
        assist_window = (
            (phase >= assist_start)
            & (phase < self.cfg.rewards.flight_start_phase)
            & (~self.jump_assisted)
        )
        use_assistance = (
            self.cfg.domain_rand.jump_start_assistance
            and not getattr(self.cfg.env, "test", False)
            and torch.any(assist_window)
        )
        if use_assistance:
            probability = self._assist_probability()
            selected = assist_window & (
                torch.rand(self.num_envs, device=self.device) < probability
            )
            if torch.any(selected):
                self.root_states[selected, 2] += (
                    self.cfg.domain_rand.jump_assist_clearance
                )
                self.root_states[selected, 7] = torch.maximum(
                    self.root_states[selected, 7],
                    torch.full_like(
                        self.root_states[selected, 7],
                        self.cfg.domain_rand.jump_assist_forward_velocity,
                    ),
                )
                self.root_states[selected, 9] = torch.maximum(
                    self.root_states[selected, 9],
                    torch.full_like(
                        self.root_states[selected, 9],
                        self.cfg.domain_rand.jump_assist_vertical_velocity,
                    ),
                )
                self.gym.set_actor_root_state_tensor(
                    self.sim, gymtorch.unwrap_tensor(self.root_states)
                )
            self.jump_assisted[assist_window] = True
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        high_enough = (
            self.root_states[:, 2]
            > self.cfg.rewards.base_height_target
            + self.cfg.rewards.minimum_flight_clearance
        )
        airborne = (~contact).all(dim=1) & high_enough
        self.airborne_now = airborne
        self.continuous_air_time = torch.where(
            airborne,
            self.continuous_air_time + self.dt,
            torch.zeros_like(self.continuous_air_time),
        )
        self.jump_max_air_time = torch.maximum(
            self.jump_max_air_time, self.continuous_air_time
        )
        self.jump_was_airborne |= airborne
        # Bootstrap the side flip: at the first airborne step, top the world
        # roll rate up toward the assist target.  The sign follows whatever
        # direction the policy already committed to, so it does not fight
        # the policy's own flip direction.
        roll_assist_window = airborne & (~self.roll_assisted)
        use_roll_assistance = (
            self.cfg.domain_rand.jump_start_assistance
            and not getattr(self.cfg.env, "test", False)
            and torch.any(roll_assist_window)
        )
        if use_roll_assistance:
            probability = self._assist_probability()
            selected = roll_assist_window & (
                torch.rand(self.num_envs, device=self.device) < probability
            )
            if torch.any(selected):
                wx = self.root_states[selected, 10]
                direction = torch.where(
                    torch.abs(wx) > 0.5,
                    torch.sign(wx),
                    torch.ones_like(wx),
                )
                self.root_states[selected, 10] = direction * torch.maximum(
                    torch.abs(wx),
                    torch.full_like(
                        wx, self.cfg.domain_rand.jump_assist_roll_velocity
                    ),
                )
                self.gym.set_actor_root_state_tensor(
                    self.sim, gymtorch.unwrap_tensor(self.root_states)
                )
            self.roll_assisted[roll_assist_window] = True
        # Store the *first* touchdown event.  A per-step contact reward let the
        # old policy collect credit later even after an ugly four-foot impact.
        self.touchdown_front_first.zero_()
        self.touchdown_rear_first.zero_()
        self.touchdown_simultaneous.zero_()
        first_touch = self.jump_was_airborne & (~self.jump_has_touched) & contact.any(dim=1)
        front = contact[:, :2].any(dim=1)
        rear = contact[:, 2:].any(dim=1)
        self.touchdown_front_first |= first_touch & front & (~rear)
        self.touchdown_rear_first |= first_touch & rear & (~front)
        self.touchdown_simultaneous |= first_touch & front & rear
        self.jump_front_supported |= self.touchdown_front_first
        self.jump_has_touched |= first_touch
        landed_now = (
            self.jump_was_airborne
            & contact.all(dim=1)
            & (self._get_phase() >= self.cfg.rewards.landing_start_phase)
        )
        self.jump_has_landed |= landed_now
        # Integrate the airborne roll angle in the world frame (direction-
        # agnostic: both CW and CCW flips accumulate positive progress).
        # The gate is feet-off-ground only: the old airborne_now gate also
        # required z > 0.475, which blinded the counter during the final
        # descent -- the flip looked complete at touchdown while the body
        # was still ~40 deg tilted and crashed on its side.
        # The integrand is |omega|: the signed version let the descent-time
        # air-righting counter-rotation SUBTRACT from the accumulated flip
        # and erase the completion, which starved the whole landing reward
        # chain of gradient.  Progress must be monotone.
        world_ang_vel = quat_apply(self.base_quat, self.base_ang_vel)
        self.cum_flip_roll += (
            torch.abs(world_ang_vel[:, 0])
            * self.dt
            * (~contact).all(dim=1).float()
        )
        self.jump_max_height = torch.maximum(
            self.jump_max_height, self.root_states[:, 2]
        )

    def _get_phase(self):
        """Normalized periodic phase; the shared Go2 version never wrapped."""
        raw_phase = self.episode_length_buf.float() * self.dt / self.cfg.rewards.cycle_time
        return torch.clamp(raw_phase, 0.0, 1.0)

    def _assist_probability(self):
        """1.0 until the assist budget is spent, then linear decay to zero."""
        full = self.cfg.domain_rand.jump_assist_full_steps
        decay = self.cfg.domain_rand.jump_assist_decay_steps
        if self.common_step_counter <= full:
            return 1.0
        return max(
            0.0,
            1.0
            - (self.common_step_counter - full) / (decay - full),
        )

    def _jump_target_height(self):
        """Target trajectory: stand, preload, extend, fly, and land."""
        phase = self._get_phase()
        standing = torch.full_like(phase, self.cfg.rewards.base_height_target)
        crouching = torch.full_like(phase, self.cfg.rewards.crouch_height_target)

        crouch_alpha = torch.clamp(
            (phase - self.cfg.rewards.preload_start_phase)
            / (self.cfg.rewards.preload_end_phase - self.cfg.rewards.preload_start_phase),
            0.0,
            1.0,
        )
        target = standing + (crouching - standing) * crouch_alpha

        extend_alpha = torch.clamp(
            (phase - self.cfg.rewards.takeoff_start_phase)
            / (self.cfg.rewards.flight_start_phase - self.cfg.rewards.takeoff_start_phase),
            0.0,
            1.0,
        )
        target = torch.where(
            phase >= self.cfg.rewards.takeoff_start_phase,
            crouching + (standing - crouching) * extend_alpha,
            target,
        )

        flight = (
            (phase >= self.cfg.rewards.flight_start_phase)
            & (phase < self.cfg.rewards.flight_end_phase)
        )
        flight_phase = torch.clamp(
            (phase - self.cfg.rewards.flight_start_phase)
            / (self.cfg.rewards.flight_end_phase - self.cfg.rewards.flight_start_phase),
            0.0,
            1.0,
        )
        flight_height = standing + self.cfg.rewards.jump_height_target * torch.sin(
            torch.pi * flight_phase
        )
        target = torch.where(flight, flight_height, target)
        target = torch.where(phase >= self.cfg.rewards.flight_end_phase, standing, target)
        absorption = (
            (phase >= self.cfg.rewards.landing_start_phase)
            & (phase < self.cfg.rewards.landing_absorption_end_phase)
        )
        absorption_height = torch.full_like(
            phase, self.cfg.rewards.landing_absorb_height
        )
        target = torch.where(absorption, absorption_height, target)
        recovery_alpha = torch.clamp(
            (
                phase - self.cfg.rewards.landing_absorption_end_phase
            )
            / (
                self.cfg.rewards.landing_recovery_end_phase
                - self.cfg.rewards.landing_absorption_end_phase
            ),
            0.0,
            1.0,
        )
        recovering = (
            (phase >= self.cfg.rewards.landing_absorption_end_phase)
            & (phase < self.cfg.rewards.landing_recovery_end_phase)
        )
        recovery_height = absorption_height + (
            standing - absorption_height
        ) * recovery_alpha
        target = torch.where(recovering, recovery_height, target)
        return target

    def check_termination(self):
        base_force = torch.norm(
            self.contact_forces[:, self.termination_contact_indices, :], dim=-1
        ).amax(dim=1)
        contact = base_force > self.cfg.rewards.base_contact_force_threshold
        self.base_contact_counter = torch.where(
            contact, self.base_contact_counter + 1, torch.zeros_like(self.base_contact_counter)
        )
        self.reset_buf = (
            self.base_contact_counter >= self.cfg.rewards.base_contact_sustain_steps
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, "base_contact_counter"):
            self.base_contact_counter[env_ids] = 0
            self.jump_has_touched[env_ids] = False
            self.jump_front_supported[env_ids] = False
            self.touchdown_front_first[env_ids] = False
            self.touchdown_rear_first[env_ids] = False
            self.touchdown_simultaneous[env_ids] = False
        if hasattr(self, "continuous_air_time"):
            self.continuous_air_time[env_ids] = 0.0
        if hasattr(self, "jump_max_air_time"):
            self.jump_max_air_time[env_ids] = 0.0
        if hasattr(self, "jump_was_airborne"):
            self.jump_was_airborne[env_ids] = False
            self.jump_has_landed[env_ids] = False
            self.jump_max_height[env_ids] = self.root_states[env_ids, 2]
        if hasattr(self, "jump_assisted"):
            self.jump_assisted[env_ids] = False
            self.roll_assisted[env_ids] = False
            self.cum_flip_roll[env_ids] = 0.0
        self._set_forward_command(env_ids)

    def _reward_base_height(self):
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )
        return torch.exp(-torch.abs(base_height - self._jump_target_height()) * 10.0)

    def _reward_preload_pose(self):
        phase = self._get_phase()
        active = (
            (phase >= self.cfg.rewards.preload_start_phase)
            & (phase < self.cfg.rewards.preload_end_phase)
        )
        error = torch.mean(torch.abs(self.dof_pos - self.crouch_dof_pos), dim=1)
        return torch.exp(-5.0 * error) * active.float()

    def _reward_preload_height(self):
        phase = self._get_phase()
        active = (
            (phase >= self.cfg.rewards.preload_start_phase)
            & (phase < self.cfg.rewards.preload_end_phase)
        )
        error = torch.abs(
            self.root_states[:, 2] - self.cfg.rewards.crouch_height_target
        )
        return torch.exp(-12.0 * error) * active.float()

    def _reward_takeoff_pose(self):
        # Ends at flight_start_phase: the full leg lockout is the push-off,
        # and it must not fight the flight tuck that starts at liftoff.
        phase = self._get_phase()
        active = (
            (phase >= self.cfg.rewards.takeoff_start_phase)
            & (phase < self.cfg.rewards.flight_start_phase)
        )
        error = torch.mean(
            torch.abs(self.dof_pos - self.takeoff_dof_pos), dim=1
        )
        return torch.exp(-4.0 * error) * active.float()

    def _reward_jump_height(self):
        phase = self._get_phase()
        flight = (
            (phase >= self.cfg.rewards.flight_start_phase)
            & (phase < self.cfg.rewards.flight_end_phase)
        )
        clearance = torch.clamp(
            (self.root_states[:, 2] - self.cfg.rewards.base_height_target)
            / self.cfg.rewards.jump_height_target,
            0.0,
            1.0,
        )
        return clearance * flight.float()

    def _reward_takeoff_velocity(self):
        """Dense vertical velocity shaping in the takeoff window."""
        phase = self._get_phase()
        takeoff = (
            phase >= self.cfg.rewards.takeoff_start_phase
        ) & (phase < self.cfg.rewards.flight_start_phase + 0.04)
        progress = torch.clamp(
            self.base_lin_vel[:, 2] / self.cfg.rewards.takeoff_velocity_target,
            0.0,
            1.0,
        )
        return progress * takeoff.float()

    def _reward_forward_takeoff(self):
        phase = self._get_phase()
        active = (
            phase >= self.cfg.rewards.takeoff_start_phase
        ) & (phase < self.cfg.rewards.flight_end_phase)
        forward_progress = torch.clamp(
            self.base_lin_vel[:, 0] / self.cfg.rewards.forward_velocity_target,
            0.0,
            1.0,
        )
        vertical_progress = torch.clamp(
            self.base_lin_vel[:, 2] / self.cfg.rewards.takeoff_velocity_target,
            0.0,
            1.0,
        )
        lateral_stability = torch.exp(-4.0 * torch.square(self.base_lin_vel[:, 1]))
        return (
            forward_progress
            * vertical_progress
            * lateral_stability
            * active.float()
        )

    def _reward_real_air_time(self):
        """Reward measured continuous flight rather than scheduled flight alone."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        high_enough = (
            self.root_states[:, 2]
            > self.cfg.rewards.base_height_target
            + self.cfg.rewards.minimum_flight_clearance
        )
        airborne = (~contact).all(dim=1) & high_enough
        progress = torch.clamp(
            self.continuous_air_time / self.cfg.rewards.target_air_time,
            0.0,
            1.0,
        )
        return progress * airborne.float()

    def _jump_height_quality(self):
        """Continuous gate that is zero for a hop and one for a real jump."""
        clearance = (
            self.jump_max_height - self.cfg.rewards.base_height_target
        )
        return torch.clamp(
            (
                clearance - self.cfg.rewards.minimum_flight_clearance
            )
            / (
                self.cfg.rewards.successful_jump_clearance
                - self.cfg.rewards.minimum_flight_clearance
            ),
            0.0,
            1.0,
        )

    def _successful_jump(self):
        height_ok = (
            self.jump_max_height
            >= self.cfg.rewards.base_height_target
            + self.cfg.rewards.successful_jump_clearance
        )
        air_time_ok = (
            self.jump_max_air_time
            >= self.cfg.rewards.successful_jump_air_time
        )
        # Softened from 0.95: the policy's cumulative roll barely topped 2*pi
        # in most episodes, so the gate was almost never satisfied and the
        # landing rewards paid nothing, leaving no gradient for the landing.
        flip_ok = (
            torch.abs(self.cum_flip_roll)
            >= 0.90 * self.cfg.rewards.side_flip_roll_target
        )
        return height_ok & air_time_ok & flip_ok

    def _flight_leg_target(self):
        """Blend into the reference pounce shape, then reach for touchdown."""
        phase = self._get_phase()
        start = self.cfg.rewards.flight_tuck_start_phase
        tuck_full = self.cfg.rewards.flight_compactness_start_phase
        tuck_amount = torch.clamp(
            (phase - start) / (tuck_full - start), 0.0, 1.0
        ).unsqueeze(1)
        target = (
            self.default_dof_pos
            + tuck_amount * (self.flight_tuck_dof_pos - self.default_dof_pos)
        )
        phase_alpha = torch.clamp(
            (phase - self.cfg.rewards.landing_pose_start_phase)
            / (
                self.cfg.rewards.landing_pose_full_phase
                - self.cfg.rewards.landing_pose_start_phase
            ),
            0.0,
            1.0,
        )
        height_alpha = torch.clamp(
            (
                self.cfg.rewards.landing_extension_start_height
                - self.root_states[:, 2]
            )
            / (
                self.cfg.rewards.landing_extension_start_height
                - self.cfg.rewards.landing_extension_full_height
            ),
            0.0,
            1.0,
        )
        descending = (
            (self.root_states[:, 9] < 0.0)
            & self.jump_was_airborne
            & (~self.jump_has_landed)
        )
        height_alpha *= descending.float()
        landing_alpha = torch.maximum(phase_alpha, height_alpha).unsqueeze(1)
        target = target + landing_alpha * (self.landing_dof_pos - target)
        # Once the forefeet catch the ground, yield the front legs and reach
        # down with the rear legs instead of holding the approach pose.
        support = self.jump_front_supported & (~self.jump_has_landed)
        return torch.where(support.unsqueeze(1), self.front_support_dof_pos, target)

    def _reward_flight_leg_pose(self):
        phase = self._get_phase()
        active = (
            (phase >= self.cfg.rewards.flight_tuck_start_phase)
            & (phase < self.cfg.rewards.flight_tuck_end_phase)
            & (~self.jump_has_landed)
        )
        pose_error = torch.mean(
            torch.abs(self.dof_pos - self._flight_leg_target()), dim=1
        )
        # Softened from exp(-4*err): the deep tuck requires a fast 1.2 rad
        # swing right after liftoff, and the sharp kernel paid ~nothing for
        # partial progress, leaving no gradient path into the tuck.
        return (
            torch.exp(-2.0 * pose_error)
            * active.float()
            * self._jump_height_quality()
        )

    def _reward_landing_position(self):
        """Reward the requested landing point and penalize overshoot as well."""
        displacement_x = self.root_states[:, 0] - self.env_origins[:, 0]
        x_error = torch.abs(
            displacement_x - self.cfg.rewards.landing_distance_target
        )
        y_error = torch.abs(self.root_states[:, 1] - self.env_origins[:, 1])
        position_score = torch.clamp(1.0 - x_error / 0.40, 0.0, 1.0)
        lateral_score = torch.clamp(1.0 - y_error / 0.20, 0.0, 1.0)
        return (
            position_score
            * lateral_score
            * self.jump_has_landed.float()
            * self._successful_jump().float()
        )

    def _reward_landing_pose(self):
        phase = self._get_phase()
        landing_alpha = torch.max(
            torch.abs(self._flight_leg_target() - self.flight_tuck_dof_pos), dim=1
        ).values
        active = (
            self.jump_was_airborne
            & (~self.jump_has_landed)
            # Only the first flight counts (see landing_foot_level).
            & (~self.jump_has_touched)
            & self.airborne_now
            & (phase >= self.cfg.rewards.landing_pose_start_phase)
            & (landing_alpha > 0.05)
        )
        error = torch.mean(
            torch.abs(self.dof_pos - self.landing_dof_pos), dim=1
        )
        return (
            torch.clamp(1.0 - error / 1.20, 0.0, 1.0)
            * active.float()
            * self._jump_height_quality()
        )

    def _reward_touchdown_state(self):
        """Approach the ground level, slowly descending, roll near 2*pi."""
        active = (
            self.jump_was_airborne
            & (~self.jump_has_landed)
            # Only the first flight counts (see landing_foot_level).
            & (~self.jump_has_touched)
            & self.airborne_now
            & (self.root_states[:, 9] < 0.0)
            & (
                self.root_states[:, 2]
                < self.cfg.rewards.landing_extension_start_height
            )
        )
        vx_error = torch.abs(
            self.root_states[:, 7]
            - self.cfg.rewards.touchdown_forward_velocity_target
        )
        vz_error = torch.abs(
            self.root_states[:, 9]
            - self.cfg.rewards.touchdown_vertical_velocity_target
        )
        pitch_error = torch.abs(self.base_euler_xyz[:, 1])
        pitch_rate = torch.abs(self.base_ang_vel[:, 1])
        # Touchdown should happen right at the completed flip: both over-
        # rotation (e.g. 450 deg) and under-rotation lose credit here.
        roll_error = torch.abs(
            torch.abs(self.cum_flip_roll)
            - self.cfg.rewards.side_flip_roll_target
        )
        # Residual spin at touchdown tips the robot over sideways; the policy
        # must extend the legs early enough to bleed the roll rate.
        world_ang_vel = quat_apply(self.base_quat, self.base_ang_vel)
        roll_rate = torch.abs(world_ang_vel[:, 0])
        # The body must actually be level in roll at touchdown.  The
        # cum-based error cannot see attitude drift.
        roll_attitude = torch.abs(self.base_euler_xyz[:, 0])
        # Average independent scores instead of multiplying them.  The former
        # product was nearly zero whenever any one component was poor.
        # The roll-rate term is softened and counted twice: it is the key
        # signal for active air righting.
        score = (
            torch.exp(-vx_error / 0.60)
            + torch.exp(-vz_error / 0.80)
            + torch.clamp(1.0 - pitch_error / 0.70, 0.0, 1.0)
            + torch.exp(-pitch_rate / 2.0)
            + torch.exp(-roll_error / 0.60)
            + 2.0 * torch.exp(-roll_rate / 4.0)
            + torch.exp(-roll_attitude / 0.40)
        ) / 8.0
        return score * active.float() * self._jump_height_quality()

    def _reward_flight_pitch(self):
        """Directly penalize nose-up/down motion after a real takeoff."""
        active = self.jump_was_airborne & (~self.jump_has_landed)
        pitch = self.base_euler_xyz[:, 1]
        pitch_rate = self.base_ang_vel[:, 1]
        error = torch.square(pitch) + 0.10 * torch.square(pitch_rate)
        return error * active.float() * self._jump_height_quality()

    def _reward_takeoff_pitch_rate(self):
        """Prevent rear-leg thrust from injecting nose-down angular momentum."""
        phase = self._get_phase()
        active = (
            (phase >= self.cfg.rewards.takeoff_start_phase)
            & (phase < self.cfg.rewards.flight_start_phase + 0.04)
        )
        return torch.square(self.base_ang_vel[:, 1]) * active.float()

    def _reward_takeoff_roll_rate(self):
        """Reward roll angular momentum about the world x-axis at takeoff."""
        phase = self._get_phase()
        active = (
            (phase >= self.cfg.rewards.takeoff_start_phase - 0.04)
            & (phase < self.cfg.rewards.flight_start_phase + 0.06)
        )
        world_ang_vel = quat_apply(self.base_quat, self.base_ang_vel)
        error = (
            torch.abs(world_ang_vel[:, 0])
            - self.cfg.rewards.takeoff_roll_rate_target
        )
        return torch.exp(-torch.square(error) / 9.0) * active.float()

    def _reward_side_flip_progress(self):
        """Dense progress toward one full lateral flip, direction-agnostic."""
        active = (
            self.jump_was_airborne
            & (~self.jump_has_landed)
            & self.airborne_now
        )
        progress = torch.clamp(
            torch.abs(self.cum_flip_roll)
            / self.cfg.rewards.side_flip_roll_target,
            0.0,
            1.0,
        )
        return progress * active.float() * self._jump_height_quality()

    def _reward_landing_foot_level(self):
        """Bring all four foot tips to the ground plane together on descent."""
        phase = self._get_phase()
        active = (
            self.jump_was_airborne
            & (~self.jump_has_landed)
            # Only the first flight counts: after a crash touchdown the
            # policy learned to hop again and re-harvest this reward.
            & (~self.jump_has_touched)
            & self.airborne_now
            & (phase >= self.cfg.rewards.landing_pose_start_phase)
        )
        foot_z = self.rigid_state[:, self.feet_indices, 2]
        # Side flip: all four feet should arrive level (offset 0).  The old
        # pounce task wanted the front feet 10 cm lower.
        front_rear_offset = (
            foot_z[:, :2].mean(dim=1) - foot_z[:, 2:].mean(dim=1)
        )
        front_rear_error = torch.abs(
            front_rear_offset - self.cfg.rewards.landing_front_rear_offset
        )
        left_right_error = 0.5 * (
            torch.abs(foot_z[:, 0] - foot_z[:, 1])
            + torch.abs(foot_z[:, 2] - foot_z[:, 3])
        )
        error = front_rear_error + 0.5 * left_right_error
        return (
            torch.clamp(1.0 - error / 0.15, 0.0, 1.0)
            * active.float()
            * self._jump_height_quality()
        )

    def _reward_simultaneous_touchdown(self):
        """Penalize the exact stiff four-foot impact seen in the recordings."""
        return (
            self.touchdown_simultaneous.float()
            * self._jump_height_quality()
        )

    def _reward_landing_impact(self):
        forces = torch.norm(
            self.contact_forces[:, self.feet_indices, :], dim=-1
        )
        excess = torch.clamp(
            (
                forces
                - self.cfg.rewards.landing_impact_force_threshold
            )
            / self.cfg.rewards.landing_impact_force_threshold,
            0.0,
            2.0,
        )
        return excess.mean(dim=1) * self.jump_was_airborne.float()

    def _reward_landing_absorption_pose(self):
        """Bend all four legs after touchdown instead of landing as stiff rods."""
        phase = self._get_phase()
        active = (
            self.jump_has_landed
            & (phase < self.cfg.rewards.landing_absorption_end_phase)
        )
        pose_error = torch.mean(
            torch.abs(self.dof_pos - self.landing_absorb_dof_pos), dim=1
        )
        # NOT gated on _successful_jump(): the absorb is landing shaping and
        # must pay after any real four-foot landing.  The old flip gate kept
        # it at exactly zero, so the policy never received a landing gradient.
        return (
            torch.clamp(1.0 - pose_error / 1.20, 0.0, 1.0)
            * active.float()
            * self._jump_height_quality()
        )

    def _reward_landing_spin(self):
        """Strong penalty for residual roll spin in the final descent.

        Angular momentum is conserved in flight, so after the tuck-untuck
        cycle the spin at touchdown is roughly the takeoff spin (~9-10
        rad/s).  The only way below that is active counter-rotation with
        the legs (air righting), which needs a dedicated gradient.
        """
        active = (
            self.jump_was_airborne
            & self.airborne_now
            & (self.root_states[:, 9] < 0.0)
            & (self.root_states[:, 2] < 0.75)
        )
        world_ang_vel = quat_apply(self.base_quat, self.base_ang_vel)
        return (
            torch.abs(world_ang_vel[:, 0])
            * active.float()
            * self._jump_height_quality()
        )

    def _reward_landing_stability(self):
        phase = self._get_phase()
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        lin_stable = torch.exp(-torch.sum(torch.square(self.base_lin_vel), dim=1) / 0.4)
        ang_stable = torch.exp(-torch.sum(torch.square(self.base_ang_vel), dim=1) / 1.5)
        upright = torch.exp(-5.0 * torch.norm(self.projected_gravity[:, :2], dim=1))
        height_error = torch.abs(
            self.root_states[:, 2] - self.cfg.rewards.base_height_target
        )
        return (
            lin_stable
            * ang_stable
            * upright
            * torch.exp(-8.0 * height_error)
            * contact.all(dim=1).float()
            * self.jump_has_landed.float()
            * self._successful_jump().float()
            * (
                phase >= self.cfg.rewards.landing_absorption_end_phase
            ).float()
        )
